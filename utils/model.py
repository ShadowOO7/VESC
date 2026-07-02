"""
model.py — Full multimodal entity reranker architecture

Components
──────────
MMoEEncoder          CLIP backbone → (text_cls, image_cls, text_tokens, image_tokens)
BiCrossAttention     Bidirectional text ↔ image cross-attention
DynamicVectorGatedFusion  Feature-wise + sample-wise gated fusion
ImageConfidenceGate  Learned scalar gate: should we trust the entity image?
MMoEMatcher          Global (CLS) + local (token) scoring with confidence-gated entity embeds
EntityReranker       Orchestrates encode → score for B mentions × N candidates
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class MMoEEncoder(nn.Module):
    """
    Wraps CLIP and projects image features to a shared dimension dv.

    Returns
    ───────
    text_cls    [B, D_text]      — CLIP text [CLS] embedding
    image_cls   [B, dv]          — projected CLIP image embedding
    text_tokens [B, T_len, D_text]  — full token sequence
    image_tokens[B, P+1, dv]     — projected patch sequence (includes CLS patch)
    """

    def __init__(self, cfg):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(cfg.pretrained_model)
        self.image_cls_fc    = nn.Linear(cfg.input_hidden_dim,       cfg.dv)
        self.image_tokens_fc = nn.Linear(cfg.input_image_hidden_dim, cfg.dv)

    def enable_gradient_checkpointing(self):
        """
        Recomputes activations during backward instead of storing them.
        Cuts CLIP activation memory ~4× at ~20% extra compute cost.
        Call this when training with a non-zero encoder_lr.
        """
        self.clip.text_model.encoder.gradient_checkpointing_enable()
        self.clip.vision_model.encoder.gradient_checkpointing_enable()

    def freeze_backbone(self):
        """Freeze all CLIP parameters. Projection layers stay trainable."""
        for p in self.clip.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        out = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        text_cls     = out.text_embeds                    # [B, D_text]
        image_cls    = out.image_embeds                   # [B, D_image]
        text_tokens  = out.text_model_output[0]           # [B, T_len, D_text]
        image_tokens = out.vision_model_output[0]         # [B, P+1, D_image]

        image_cls    = self.image_cls_fc(image_cls)       # [B, dv]
        image_tokens = self.image_tokens_fc(image_tokens) # [B, P+1, dv]

        return text_cls, image_cls, text_tokens, image_tokens


# ─────────────────────────────────────────────────────────────────────────────
# BI-DIRECTIONAL CROSS ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class BiCrossAttention(nn.Module):
    """
    text_tokens attends to image_tokens  (t2v)
    image_tokens attends to text_tokens  (v2t)
    Both paths residual + layer-normed.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.norm_text  = nn.LayerNorm(hidden_dim)
        self.norm_image = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_tokens:  torch.Tensor,   # [B, T, D]
        image_tokens: torch.Tensor,   # [B, P, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        t2v, _ = self.text_to_image(text_tokens,  image_tokens, image_tokens)
        v2t, _ = self.image_to_text(image_tokens, text_tokens,  text_tokens)

        text_tokens  = self.norm_text (text_tokens  + t2v)
        image_tokens = self.norm_image(image_tokens + v2t)

        return text_tokens, image_tokens


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC VECTOR GATED FUSION
# ─────────────────────────────────────────────────────────────────────────────

class DynamicVectorGatedFusion(nn.Module):
    """
    Combines two vectors via:
      1. Feature-wise gate  (gate_fc)   — which dimensions to keep from each
      2. Sample-wise blend  (lambda_fc) — how much fused vs mean fallback
      3. Learned temperature scaling
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lambda_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([feat1, feat2], dim=-1)          # [B, 2D]

        gate   = torch.sigmoid(self.gate_fc(concat))        # [B, D]
        lam    = torch.sigmoid(self.lambda_fc(concat))      # [B, 1]

        fused  = gate * feat1 + (1 - gate) * feat2
        fused  = lam  * fused + (1 - lam)  * (feat1 + feat2) / 2
        # Note: no temperature scaling here — output is L2-normalised downstream
        # so any scalar factor would cancel out and only destabilise gradients.
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE CONFIDENCE GATE
# ─────────────────────────────────────────────────────────────────────────────

class ImageConfidenceGate(nn.Module):
    """
    Learns a scalar confidence ∈ [0, 1] expressing how useful the
    entity image is given the text context.

        confidence = 0  →  ignore image, fall back to text-only embedding
        confidence = 1  →  fully use the multimodal fused embedding

    Architecture:  Linear(2D → D) → ReLU → Linear(D → 1) → Sigmoid
    Input:  concat(text_vec, image_vec)  [B, 2D]
    Output: confidence                   [B, 1]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, text_vec: torch.Tensor, image_vec: torch.Tensor
    ) -> torch.Tensor:
        return self.gate(torch.cat([text_vec, image_vec], dim=-1))  # [B, 1]


# ─────────────────────────────────────────────────────────────────────────────
# MATCHER
# ─────────────────────────────────────────────────────────────────────────────

class MMoEMatcher(nn.Module):
    """
    Scores a (mention, entity) pair using two complementary signals.

    Global path  (text CLS)
      mention_global = text_cls_proj(mention_text_cls)
      entity_global  = text_cls_proj(entity_text_cls)
      global_score   = cosine similarity

    Local path  (token-level with cross-attention)
      Mention  → text-only embedding  (image_tokens=None)
      Entity   → confidence-gated multimodal embedding

    Combined score = alpha * global + beta * local
    """

    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_dim

        self.text_proj  = nn.Linear(cfg.input_hidden_dim, D)
        self.image_proj = nn.Linear(cfg.dv, D)

        self.cross_layers = nn.ModuleList([
            BiCrossAttention(D) for _ in range(cfg.cross_layers)
        ])

        self.fusion               = DynamicVectorGatedFusion(D)
        self.image_confidence_gate = ImageConfidenceGate(D)
        self.text_cls_proj        = nn.Linear(cfg.dt, D)

        # alpha/beta are NOT fixed — call set_alpha_beta() each epoch to
        # warm up the local head gradually.  Start global-heavy, end local-heavy.
        self.alpha = 0.9   # global weight  (start high, local head is random at init)
        self.beta  = 0.1   # local weight   (ramp up as training progresses)

    def set_alpha_beta(self, alpha: float):
        """Schedule: call with alpha decreasing from 0.9 → 0.3 over epochs."""
        self.alpha = max(0.3, alpha)
        self.beta  = 1.0 - self.alpha

    # ── Core embedding function ───────────────────────────────────────────────

    def multimodal_embedding(
        self,
        text_tokens:  torch.Tensor,
        image_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        image_tokens=None  → text-only path   (used for mentions)
        image_tokens≠None  → confidence-gated multimodal path  (used for entities)

        Returns L2-normalised embedding  [B, D].
        """
        text_tokens = self.text_proj(text_tokens)       # [B, T, D]

        # ── Text-only (mention) ───────────────────────────────────
        if image_tokens is None:
            return F.normalize(text_tokens.mean(dim=1), dim=-1)

        # ── Multimodal (entity) ───────────────────────────────────
        image_tokens = self.image_proj(image_tokens)    # [B, P, D]

        for layer in self.cross_layers:
            text_tokens, image_tokens = layer(text_tokens, image_tokens)

        text_vec  = text_tokens.mean(dim=1)             # [B, D]
        image_vec = image_tokens.mean(dim=1)            # [B, D]

        # Branch 1: fused multimodal representation
        multimodal_embed = self.fusion(text_vec, image_vec)     # [B, D]

        # Branch 2: text-only fallback (no image influence)
        text_only_embed  = text_vec                              # [B, D]

        # Confidence gate: how much to trust the image for this entity
        confidence = self.image_confidence_gate(text_vec, image_vec)  # [B, 1]

        embed = confidence * multimodal_embed + (1 - confidence) * text_only_embed
        return F.normalize(embed, dim=-1)

    # ── Original cross-batch forward (kept for in-batch contrastive training) ─

    def forward(
        self,
        entity_text_cls,    entity_text_tokens,
        mention_text_cls,   mention_text_tokens,
        entity_image_cls,   entity_image_tokens,
        mention_image_cls=None, mention_image_tokens=None,
    ):
        # Global
        m_global = F.normalize(self.text_cls_proj(mention_text_cls), dim=-1)
        e_global = F.normalize(self.text_cls_proj(entity_text_cls),  dim=-1)
        global_score = torch.matmul(m_global, e_global.T)

        # Local
        m_embed = self.multimodal_embedding(mention_text_tokens, None)
        e_embed = self.multimodal_embedding(entity_text_tokens, entity_image_tokens)
        local_score = torch.matmul(m_embed, e_embed.T)

        score = self.alpha * global_score + self.beta * local_score
        return score, global_score, local_score


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY RERANKER  (top-level module used during training & inference)
# ─────────────────────────────────────────────────────────────────────────────

class EntityReranker(nn.Module):
    """
    Encodes B mentions and their B×N candidate entities, then scores
    each candidate against its corresponding mention.

    Input batch keys
    ────────────────
    m_input_ids        [B, T]
    m_attention_mask   [B, T]
    m_pixel_values     [B, 3, H, W]
    e_input_ids        [B, N, T]
    e_attention_mask   [B, N, T]
    e_pixel_values     [B, N, 3, H, W]

    Returns
    ───────
    scores         [B, N]  — combined alpha*global + beta*local
    global_scores  [B, N]
    local_scores   [B, N]
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.encoder = MMoEEncoder(model_cfg)
        self.matcher = MMoEMatcher(model_cfg)
        self._mini_batch = model_cfg.__dict__.get("encoder_mini_batch", 64)

    # ── Mini-batch encoding (avoids OOM on B×N) ───────────────────────────────

    def _encode(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder in chunks of self._mini_batch rows."""
        total = input_ids.shape[0]
        if total <= self._mini_batch:
            return self.encoder(input_ids, attention_mask, pixel_values)

        parts = [[], [], [], []]
        for start in range(0, total, self._mini_batch):
            end = min(start + self._mini_batch, total)
            out = self.encoder(
                input_ids[start:end],
                attention_mask[start:end],
                pixel_values[start:end],
            )
            for i, t in enumerate(out):
                parts[i].append(t)

        return tuple(torch.cat(p, dim=0) for p in parts)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict):
        B = batch["m_input_ids"].shape[0]
        N = batch["e_input_ids"].shape[1]

        # ── 1. Encode mention ─────────────────────────────────────
        m_text_cls, _, m_text_tokens, _ = self.encoder(
            input_ids=batch["m_input_ids"],
            attention_mask=batch["m_attention_mask"],
            pixel_values=batch["m_pixel_values"],
        )
        # m_text_cls:    [B, D_text]
        # m_text_tokens: [B, T_len, D_text]

        # ── 2. Encode entities  (flatten B×N → B*N, encode, reshape) ─────────
        H, W   = batch["e_pixel_values"].shape[-2:]
        e_ids  = batch["e_input_ids"].view(B * N, -1)
        e_mask = batch["e_attention_mask"].view(B * N, -1)
        e_pix  = batch["e_pixel_values"].view(B * N, 3, H, W)

        e_text_cls, _, e_text_tokens, e_image_tokens = self._encode(
            e_ids, e_mask, e_pix
        )
        # e_text_cls:     [B*N, D_text]
        # e_text_tokens:  [B*N, T_len, D_text]
        # e_image_tokens: [B*N, P+1, dv]

        # ── 3. Global scores via text CLS ─────────────────────────
        m_global = F.normalize(self.matcher.text_cls_proj(m_text_cls), dim=-1)   # [B, D]
        e_global = F.normalize(self.matcher.text_cls_proj(e_text_cls), dim=-1)   # [B*N, D]
        e_global = e_global.view(B, N, -1)                                         # [B, N, D]

        # bmm: (B, 1, D) × (B, D, N) → (B, 1, N) → (B, N)
        global_scores = torch.bmm(
            m_global.unsqueeze(1),
            e_global.transpose(1, 2),
        ).squeeze(1)                                                                # [B, N]

        # ── 4. Local scores via token embeddings ──────────────────
        m_embed = self.matcher.multimodal_embedding(m_text_tokens, None)          # [B, D]
        e_embed = self.matcher.multimodal_embedding(e_text_tokens, e_image_tokens) # [B*N, D]
        e_embed = e_embed.view(B, N, -1)                                            # [B, N, D]

        local_scores = torch.bmm(
            m_embed.unsqueeze(1),
            e_embed.transpose(1, 2),
        ).squeeze(1)                                                                # [B, N]

        # ── 5. Combined score ─────────────────────────────────────
        scores = self.matcher.alpha * global_scores + self.matcher.beta * local_scores

        return scores, global_scores, local_scores


# ─────────────────────────────────────────────────────────────────────────────
# CACHED ENTITY RERANKER  (uses precomputed CLIP features — no CLIP backbone)
# ─────────────────────────────────────────────────────────────────────────────

class CachedEntityReranker(nn.Module):
    """
    Drop-in replacement for EntityReranker that skips the CLIP forward pass.

    Instead of running MMoEEncoder end-to-end, this module:
      1. Applies the trainable projection layers that sit inside MMoEEncoder
         (image_cls_fc, image_tokens_fc) to the precomputed raw CLIP features.
      2. Passes the projected features directly to MMoEMatcher.

    All matcher weights (cross-attention, fusion, confidence gate, CLS proj)
    are identical to EntityReranker — a checkpoint saved from either class
    can be loaded by the other as long as state_dict keys match.

    Expected batch keys (from CachedCandidateDataset / cached_collate_fn)
    ───────────────────────────────────────────────────────────────────────
      m_text_cls        [B, D_text]
      m_text_tokens     [B, T, D_text]
      e_text_cls        [B, N, D_text]
      e_text_tokens     [B, N, T, D_text]
      e_image_tokens    [B, N, P+1, D_vision]
      label_idx         [B]  (training only)
    """

    def __init__(self, model_cfg):
        super().__init__()

        # Projection layers from MMoEEncoder (trainable, not precomputed)
        self.image_cls_fc    = nn.Linear(model_cfg.input_hidden_dim,       model_cfg.dv)
        self.image_tokens_fc = nn.Linear(model_cfg.input_image_hidden_dim, model_cfg.dv)

        # Full matcher (cross-attention, fusion, confidence gate, scoring)
        self.matcher = MMoEMatcher(model_cfg)

    # ── Checkpoint compatibility ───────────────────────────────────────────────

    @classmethod
    def from_entity_reranker_checkpoint(
        cls,
        ckpt_path:  str,
        model_cfg,
        map_location: str = "cpu",
    ) -> "CachedEntityReranker":
        """
        Load weights saved from an EntityReranker checkpoint.

        The key mapping is:
          encoder.image_cls_fc.*    →  image_cls_fc.*
          encoder.image_tokens_fc.* →  image_tokens_fc.*
          matcher.*                 →  matcher.*  (unchanged)
        """
        ckpt       = torch.load(ckpt_path, map_location=map_location)
        full_state = ckpt["model"]

        remapped = {}
        for k, v in full_state.items():
            if k.startswith("encoder.image_cls_fc."):
                remapped[k.replace("encoder.", "", 1)] = v
            elif k.startswith("encoder.image_tokens_fc."):
                remapped[k.replace("encoder.", "", 1)] = v
            elif k.startswith("matcher."):
                remapped[k] = v
            # encoder.clip.* is intentionally dropped

        model = cls(model_cfg)
        missing, unexpected = model.load_state_dict(remapped, strict=False)

        if missing:
            raise RuntimeError(f"Missing keys when loading cached reranker: {missing}")

        return model

    def to_entity_reranker_state(self) -> dict:
        """
        Return a state dict compatible with EntityReranker so you can
        resume full (non-cached) training if needed.

        Only the projection and matcher weights are included; the caller
        must merge with a CLIP backbone state dict separately.
        """
        state = {}
        for k, v in self.image_cls_fc.state_dict().items():
            state[f"encoder.image_cls_fc.{k}"] = v
        for k, v in self.image_tokens_fc.state_dict().items():
            state[f"encoder.image_tokens_fc.{k}"] = v
        for k, v in self.matcher.state_dict().items():
            state[f"matcher.{k}"] = v
        return state

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict):
        """
        batch contains precomputed raw CLIP tensors (float32 after loading).
        We apply the projection layers here, then score via MMoEMatcher.
        """
        B = batch["m_text_cls"].shape[0]
        N = batch["e_text_cls"].shape[1]

        # ── 1. Mention features ──────────────────────────────────────
        # Mention uses text-only path — no image projection needed
        m_text_cls    = batch["m_text_cls"]      # [B, D_text]
        m_text_tokens = batch["m_text_tokens"]   # [B, T, D_text]

        # ── 2. Entity features  (flatten B×N → B*N) ─────────────────
        e_text_cls    = batch["e_text_cls"].view(B * N, -1)                    # [B*N, D_text]
        e_text_tokens = batch["e_text_tokens"].view(B * N,
                            batch["e_text_tokens"].shape[2], -1)               # [B*N, T, D_text]

        # Apply trainable projection to raw image patch features
        P1, D_vis    = batch["e_image_tokens"].shape[2], batch["e_image_tokens"].shape[3]
        e_image_raw  = batch["e_image_tokens"].view(B * N, P1, D_vis)         # [B*N, P+1, D_vis]
        e_image_tokens = self.image_tokens_fc(e_image_raw)                     # [B*N, P+1, dv]

        # ── 3. Global scores (text CLS) ──────────────────────────────
        m_global = F.normalize(self.matcher.text_cls_proj(m_text_cls), dim=-1)   # [B, D]
        e_global = F.normalize(self.matcher.text_cls_proj(e_text_cls), dim=-1)   # [B*N, D]
        e_global = e_global.view(B, N, -1)                                         # [B, N, D]

        global_scores = torch.bmm(
            m_global.unsqueeze(1),
            e_global.transpose(1, 2),
        ).squeeze(1)                                                                # [B, N]

        # ── 4. Local scores (token-level cross-attention + fusion) ───
        m_embed = self.matcher.multimodal_embedding(m_text_tokens, None)          # [B, D]
        e_embed = self.matcher.multimodal_embedding(e_text_tokens, e_image_tokens) # [B*N, D]
        e_embed = e_embed.view(B, N, -1)

        local_scores = torch.bmm(
            m_embed.unsqueeze(1),
            e_embed.transpose(1, 2),
        ).squeeze(1)                                                                # [B, N]

        # ── 5. Combined score ────────────────────────────────────────
        scores = self.matcher.alpha * global_scores + self.matcher.beta * local_scores

        # ── 6. Retrieval score anchor (optional) ─────────────────────────
        # Blends in the original retrieval score so the model falls back to
        # retrieval ordering when uncertain, preventing recall@k degradation.
        if "retrieval_scores" in batch:
            ret = batch["retrieval_scores"].to(scores.device)        # [B, N]
            r_min = ret.min(dim=1, keepdim=True).values
            r_max = ret.max(dim=1, keepdim=True).values
            ret_norm = 2.0 * (ret - r_min) / (r_max - r_min + 1e-8) - 1.0
            scores = scores + 0.2 * ret_norm

        return scores, global_scores, local_scores
