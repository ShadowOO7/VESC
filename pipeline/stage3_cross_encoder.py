"""
cross_encoder.py — Stage 1 replacement: BGE cross-encoder reranker.

Why this beats the CLIP bi-encoder for this dataset
──────────────────────────────────────────────────────
CLIP bi-encoder: encodes mention and entity SEPARATELY, then scores by
cosine similarity of the two vectors. It can't see the relationship between
the words — "Brussels attacks" and "2016 Brussels bombings" score low because
"attacks" ≠ "bombings" as isolated tokens.

Cross-encoder: encodes [mention + entity] JOINTLY in one forward pass:
  [CLS] Brussels attacks [SEP] 2016 Brussels bombings. attacks of 22 March 2016 [SEP]
The model attends across the separator and directly finds the semantic bridge.
This is consistently 10-15pp better than bi-encoders on reranking benchmarks.

Why this works with bare surface mention strings
──────────────────────────────────────────────────
The entity_text is a short Wikipedia description ("EntityTitle. short desc").
When the mention is "Church of Scientology" and the entity is
"Church of Scientology. American religious cult and business", it's an
exact prefix match — trivially correct. The hard cases are ambiguous
mentions like "attacks" or "Brussels" where the entity descriptions
carry all the discriminating signal — and that's exactly what the
cross-encoder reads.

Model: BAAI/bge-reranker-v2-m3
  568M params | fp16 ≈ 1.1 GB VRAM
  State-of-the-art on BEIR benchmark, strong zero-shot performance
  No h5 cache needed — reads text from JSON directly

Expected performance
──────────────────────
Zero-shot (no fine-tuning):   Recall@1 ~76-80%,  Recall@10 ~93-95%
After fine-tuning (3 epochs): Recall@1 ~82-87%,  Recall@10 ~93-95%

This compares to the retrieval baseline (Recall@1 72.8%, Recall@10 94.3%)
and the broken CLIP Stage 1 (Recall@1 57.8%, Recall@10 60%).

Run order
──────────
# Option A — zero-shot (fastest, skip fine-tuning)
python train_pipeline.py --generate-topk

# Option B — fine-tune first (recommended, ~20 min)
python train_pipeline.py --stage 1 --phase warmup
python train_pipeline.py --generate-topk --stage1-ckpt checkpoints/cross_encoder/best/

Interface with Stage 3
───────────────────────
Produces STAGE1_TOPK format expected by LLMRerankDataset:
[{
  "mention_id": "m001",
  "label_entity_id": "Q19837",
  "ranked": [{"entity_id": "Q19837", "score": 0.91}, ...]   # top_k entries
}]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Placeholder for entities with no text at all.
# The model learns this token = low relevance signal.
EMPTY_TEXT_TOKEN = "[NO_DESCRIPTION]"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderDataset(Dataset):
    """
    One sample = one mention + all N candidate entity texts.
    Used for both fine-tuning and evaluation.

    Returns a plain dict (no tensors) — tokenisation happens inside the
    training loop so we can control the batch of N pairs per mention.
    """

    def __init__(
        self,
        json_path:      str,
        num_candidates: int  = 50,
        is_train:       bool = True,
    ):
        self.num_candidates = num_candidates
        self.is_train       = is_train

        with open(json_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        self.data = self._prepare(raw)
        logger.info(f"CrossEncoderDataset: {len(self.data)} samples from {json_path}")

    @staticmethod
    def _clean_entity_text(cand: Dict) -> str:
        t = cand.get("entity_text", "").strip().rstrip(".")
        # "Bravest. " after strip+rstrip becomes "Bravest" — still meaningful.
        # Only use the placeholder for completely empty strings.
        return t if t else EMPTY_TEXT_TOKEN

    def _prepare(self, raw: List[Dict]) -> List[Dict]:
        out, skipped = [], 0
        for rec in raw:
            cands = list(rec["candidates"])

            # Pad or trim to exactly num_candidates
            if len(cands) > self.num_candidates:
                cands = cands[: self.num_candidates]
            elif len(cands) < self.num_candidates:
                cands += [cands[-1]] * (self.num_candidates - len(cands))

            # Resolve label
            label = rec.get("label_idx")
            if self.is_train:
                if label is None:
                    for i, c in enumerate(cands):
                        if c.get("is_correct", False):
                            label = i
                            break
                if label is None or not (0 <= label < self.num_candidates):
                    skipped += 1
                    continue

            if self.is_train and label is not None:
                import random
                indices = list(range(len(cands)))
                random.shuffle(indices)
                cands = [cands[i] for i in indices]
                label = indices.index(label)
            out.append({
                "mention_id":   rec["mention_id"],
                "mention_text": rec.get("mention_text", ""),
                "entity_ids":   [c["entity_id"]                  for c in cands],
                "entity_texts": [self._clean_entity_text(c)      for c in cands],
                "label_idx":    label,
            })

        if skipped:
            logger.warning(f"Skipped {skipped} samples (missing/invalid label)")
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ENCODER RERANKER
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps BAAI/bge-reranker-v2-m3 (or any AutoModelForSequenceClassification)
    for entity candidate reranking.

    Scoring:  for each mention, tokenise all N (mention, entity) pairs as one
              batch → single forward pass → N relevance logits.
    Training: listwise CE — the N logits are treated as class scores, CE loss
              pushes the correct candidate to rank 1 while preserving relative
              ordering of the rest.

    Parameters
    ──────────
    model_name   HuggingFace model ID (default: BAAI/bge-reranker-v2-m3)
    device       "cuda" or "cpu"
    fp16         use float16 inference/training (recommended for 7B+ setups)
    max_length   max tokens per (mention, entity) pair (128 is sufficient)
    """

    def __init__(
        self,
        model_name: str  = "BAAI/bge-reranker-v2-m3",
        device:     str  = "cuda",
        fp16:       bool = True,
        max_length: int  = 128,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading {model_name}…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

        self.device     = torch.device(device)
        self.fp16       = fp16
        self.max_length = max_length
        logger.info(
            f"CrossEncoderReranker ready — "
            f"{sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params | "
            f"device={device} | fp16={fp16}"
        )

    # ── Core scoring ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_candidates(
        self,
        mention_text: str,
        entity_texts: List[str],
        inner_batch:  int = 64,
    ) -> torch.Tensor:
        """
        Score all candidates for one mention.

        Processes in inner_batch chunks to avoid OOM with N=50 long texts.
        Returns float32 tensor of shape [N] — higher = more relevant.
        """
        all_scores = []
        N = len(entity_texts)

        for start in range(0, N, inner_batch):
            end   = min(start + inner_batch, N)
            pairs = [(mention_text, et) for et in entity_texts[start:end]]

            enc = self.tokenizer(
                pairs,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with autocast("cuda", enabled=self.fp16):
                logits = self.model(**enc).logits   # [B, 1] for bge-reranker

            all_scores.append(logits[:, 0].float().cpu())

        return torch.cat(all_scores)   # [N]

    # ── File reranking → STAGE1_TOPK format ──────────────────────────────────

    def rerank_json(
        self,
        json_path:   str,
        output_path: Optional[str] = None,
        top_k:       int = 10,
    ) -> List[Dict]:
        """
        Score all 50 candidates for every mention in json_path.
        Returns (and optionally saves) the STAGE1_TOPK format that
        LLMRerankDataset in llm_reranker.py expects.
        """
        with open(json_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)

        results = []
        for rec in tqdm(records, desc=f"Reranking {Path(json_path).name}"):
            cands        = rec["candidates"]
            mention_text = rec.get("mention_text", "")
            entity_texts = [
                CrossEncoderDataset._clean_entity_text(c) for c in cands
            ]

            scores = self.score_candidates(mention_text, entity_texts)   # [N]

            # Resolve label entity id (for building Stage 3 training data)
            label_idx = rec.get("label_idx")
            if label_idx is None:
                for i, c in enumerate(cands):
                    if c.get("is_correct", False):
                        label_idx = i
                        break
            label_eid = cands[label_idx]["entity_id"] if label_idx is not None else None

            order  = scores.argsort(descending=True).tolist()[:top_k]
            ranked = [
                {"entity_id": cands[j]["entity_id"], "score": float(scores[j])}
                for j in order
            ]

            results.append({
                "mention_id":      rec["mention_id"],
                "label_entity_id": label_eid,
                "ranked":          ranked,
            })

        logger.info(f"Reranked {len(results)} mentions")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
            logger.info(f"Saved → {output_path}")

        return results

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        json_path:   str,
        top_k_list:  Tuple = (1, 5, 10),
    ) -> Dict[str, float]:
        """Rerank json_path and compute Recall@k + MRR."""
        # Use full top_k = max(top_k_list) so all k values are covered
        results  = self.rerank_json(json_path, top_k=max(top_k_list))
        counters = {k: 0 for k in top_k_list}
        rr_sum, total = 0.0, 0

        for r in results:
            if r["label_entity_id"] is None:
                continue
            ranked  = [x["entity_id"] for x in r["ranked"]]
            correct = r["label_entity_id"]
            for k in top_k_list:
                if correct in ranked[:k]:
                    counters[k] += 1
            if correct in ranked:
                rr_sum += 1.0 / (ranked.index(correct) + 1)
            total += 1

        metrics = {f"recall@{k}": counters[k] / max(total, 1) for k in top_k_list}
        metrics["mrr"] = rr_sum / max(total, 1)

        logger.info("CrossEncoder evaluation:")
        for k, v in metrics.items():
            logger.info(f"  {k:12s}: {v:.4f}")

        return metrics

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def fine_tune(
        self,
        train_json:     str,
        val_json:       str,
        output_dir:     str,
        num_epochs:     int   = 3,
        lr:             float = 2e-5,
        weight_decay:   float = 1e-2,
        warmup_ratio:   float = 0.06,
        grad_clip:      float = 1.0,
        num_candidates: int   = 50,
        log_every:      int   = 100,
    ):
        """
        Listwise fine-tuning on labeled (mention, 50 candidates) data.

        For each mention, all N (mention, entity_text) pairs are tokenised
        and passed through the model in a single forward pass.  The resulting
        N logits are treated as class scores and a CE loss pushes the correct
        candidate's score above the rest.

        This is more data-efficient than pairwise training (uses all 49
        negatives per step) and aligns directly with the recall@k metric.

        One mention = one gradient step (batch_size=1 in the DataLoader).
        Use gradient accumulation to simulate larger effective batches.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_ds = CrossEncoderDataset(train_json, num_candidates, is_train=True)
        val_ds   = CrossEncoderDataset(val_json,   num_candidates, is_train=True)

        # batch_size=1: each "batch" is one mention + its N candidates.
        # The N pairs are handled as an inner batch inside the training step.
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True,
            collate_fn=lambda b: b[0],   # unwrap the single-element list
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            collate_fn=lambda b: b[0],
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        total_steps = len(train_loader) * num_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
        )
        scaler = GradScaler("cuda", enabled=self.fp16)

        best_recall1 = 0.0
        log_path     = output_dir / "train_log.jsonl"

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss  = 0.0
            step_in_epoch = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for sample in pbar:
                mention_text = sample["mention_text"]
                entity_texts = sample["entity_texts"]
                label_idx    = sample["label_idx"]

                # Tokenise all N (mention, entity) pairs as one batch
                pairs = [(mention_text, et) for et in entity_texts]
                enc = self.tokenizer(
                    pairs,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                optimizer.zero_grad()

                with autocast("cuda", enabled=self.fp16):
                    logits = self.model(**enc).logits[:, 0]          # [N]
                    loss   = F.cross_entropy(
                        logits.unsqueeze(0),                          # [1, N]
                        torch.tensor([label_idx], device=self.device),
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss    += loss.item()
                step_in_epoch += 1

                if step_in_epoch % log_every == 0:
                    pbar.set_postfix(loss=f"{epoch_loss/step_in_epoch:.4f}")

            avg_loss = epoch_loss / max(step_in_epoch, 1)
            metrics  = self._eval_loop(val_loader)
            recall1  = metrics["recall@1"]

            log_entry = {"epoch": epoch + 1, "train_loss": avg_loss, **metrics}
            with open(log_path, "a") as fh:
                fh.write(json.dumps(log_entry) + "\n")

            logger.info(
                f"Epoch {epoch+1} | loss={avg_loss:.4f} | "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

            if recall1 > best_recall1:
                best_recall1 = recall1
                self.save(str(output_dir / "best"))
                logger.info(f"  ↑ New best Recall@1: {recall1:.4f}")

        logger.info(f"Fine-tuning complete. Best Recall@1: {best_recall1:.4f}")

    @torch.no_grad()
    def _eval_loop(
        self,
        val_loader,
        top_k_list: Tuple = (1, 5, 10),
    ) -> Dict[str, float]:
        """Run evaluation over a DataLoader of CrossEncoderDataset samples."""
        self.model.eval()
        counters = {k: 0 for k in top_k_list}
        rr_sum, total = 0.0, 0

        for sample in tqdm(val_loader, desc="  [Val]", leave=False):
            scores   = self.score_candidates(
                sample["mention_text"], sample["entity_texts"]
            )
            ranked   = scores.argsort(descending=True).tolist()
            label    = sample["label_idx"]
            rank     = ranked.index(label) + 1   # 1-indexed

            for k in top_k_list:
                if rank <= k:
                    counters[k] += 1
            rr_sum += 1.0 / rank
            total  += 1

        self.model.train()

        metrics = {f"recall@{k}": counters[k] / max(total, 1) for k in top_k_list}
        metrics["mrr"] = rr_sum / max(total, 1)
        return metrics

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model + tokenizer to path (compatible with from_pretrained)."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"CrossEncoder saved → {path}")

    def load(self, path: str):
        """Load model + tokenizer from a saved checkpoint."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            path,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        logger.info(f"CrossEncoder loaded from {path}")
