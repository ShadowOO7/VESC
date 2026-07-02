"""
llm_reranker_byme.py — Stage 3: LLM listwise reranker.
Changes vs llm_reranker.py
──────────────────────────
Fix 1: Stage 1 scores shown per candidate in prompt — gives LLM ranking signal
Fix 3: Sentence context from meta added to mention — fixes "Selton Mello" with no context
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RICH MENTION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_rich_mention(
    mention_text:    str,
    mention_caption: str,
    meta:            Dict,
) -> Dict[str, str]:
    """
    Combine raw mention_text with EntityExtractionPipeline metadata into
    a single display string that mirrors the UniMEL mention representation.

    meta keys (from structured_output_meta_*.json):
      generated_type        — entity type (person, location, ...)
      generated_description — 1-sentence Qwen description
      generated_tags        — list of image tags from RAM++/DINO
      sentence              — FIX 3: original sentence context

    Falls back gracefully to raw mention_text if meta is empty.
    """
    desc     = meta.get("generated_description", "").strip()
    etype    = meta.get("generated_type", "").strip()
    tags     = meta.get("generated_tags", [])
    tags_str = ", ".join(str(t) for t in tags[:8]) if tags else ""

    # FIX 3: Add sentence context so LLM sees "Selton Mello in a Brazilian film..."
    sentence = meta.get("sentence", "").strip()

    parts = [mention_text]
    if etype:
        parts.append(f"Type: {etype}")
    if desc:
        parts.append(f"Description: {desc}")
    if tags_str:
        parts.append(f"Image tags: {tags_str}")
    # Only add sentence if it adds new info not already in mention_text
    if sentence and sentence not in mention_text:
        parts.append(f"Context: {sentence}")

    return {
        "display_text": " | ".join(parts),
        "caption":      desc if desc else mention_caption,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER  (UniMEL T_ms style)
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    UniMEL-style T_ms prompt.

    Mention section: name + type + description + image tags + sentence context
    Entity table:   numbered list of (name: desc + score + optional image caption)
    Output:         single digit identifying the correct candidate
    """

    SYSTEM_PROMPT = (
        "You are an expert entity linking system. Given a mention and a list of "
        "entity candidates, identify which candidate best matches the mention. "
        "Consider the mention type, description, image tags, and entity summaries. "
        "Respond with only the number of the best candidate."
    )

    def build(
        self,
        mention_text:    str,
        mention_caption: str,
        candidates:      List[Dict],
        use_scores:      bool = False,
    ) -> str:
        lines = ["### Mention"]
        # mention_text is already rich: "name | Type: X | Description: Y | Image tags: Z | Context: ..."
        for part in mention_text.split(" | "):
            if part.strip():
                lines.append(part.strip())

        lines.append("\n### Entity candidates")

        for i, cand in enumerate(candidates, start=1):
            text = (
                cand.get("entity_summary", "").strip()
                or cand.get("entity_text", "").strip()
            )
            if len(text) > 200:
                text = text[:197] + "..."

            # Split entity name from description
            parts = text.split(". ", 1)
            ename = parts[0].strip() or cand.get("entity_id", "Unknown")
            edesc = parts[1].strip() if len(parts) > 1 else ""
            if len(edesc) > 180:
                edesc = edesc[:177] + "..."

            line = f"[{i}] {ename}"
            if edesc:
                line += f": {edesc}"

            # FIX 1: Add Stage 1 score so LLM knows retrieval confidence
            score = cand.get("score")
            if score is not None:
                line += f" [score: {score:.3f}]"

            if cand.get("entity_caption"):
                line += f" | Image: {cand['entity_caption'].strip()[:80]}"

            lines.append(line)

        lines.append(
            f"\n### Task"
            f"\nWhich candidate [1\u2013{len(candidates)}] best matches the mention? "
            f"Answer with ONLY the number:"
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM RERANKER
# ─────────────────────────────────────────────────────────────────────────────

class LLMReranker(nn.Module):
    """
    Qwen2.5-7B-Instruct + LoRA for listwise entity reranking.
    """

    def __init__(
        self,
        model_name_or_path: str   = "Qwen/Qwen2.5-7B-Instruct",
        lora_r:             int   = 32,
        lora_alpha:         int   = 64,
        lora_dropout:       float = 0.05,
        top_k:              int   = 10,
        load_in_8bit:       bool  = False,
    ):
        super().__init__()
        self.top_k          = top_k
        self.prompt_builder = PromptBuilder()

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {model_name_or_path}…")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_cfg = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            quantization_config=quant_cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        from peft import LoraConfig, TaskType, get_peft_model

        self.model = get_peft_model(
            self.base_model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        )
        self.model.print_trainable_parameters()
        self.digit_token_ids = self._get_digit_token_ids(top_k)

    # ── Qwen ChatML template ──────────────────────────────────────────────────

    def _apply_chat_template(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _get_digit_token_ids(self, top_k: int) -> torch.Tensor:
        ids = []
        for i in range(1, top_k + 1):
            tok = self.tokenizer(str(i), add_special_tokens=False)["input_ids"]
            ids.append(tok[-1])
        return torch.tensor(ids, dtype=torch.long)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        prompts:   List[str],
        label_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        formatted = [self._apply_chat_template(p) for p in prompts]
        enc = self.tokenizer(
            formatted, return_tensors="pt",
            padding=True, truncation=True, max_length=2048,
        )
        input_ids      = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)
        outputs        = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_logits    = outputs.logits[:, -1, :]
        digit_ids      = self.digit_token_ids.to(self.model.device)
        digit_logits   = last_logits[:, digit_ids]
        loss           = F.cross_entropy(digit_logits, label_idx.to(self.model.device))
        return {"loss": loss, "digit_logits": digit_logits.detach()}

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_candidates(
        self,
        mention_text:    str,
        mention_caption: str,
        candidates:      List[Dict],
        n_tta:           int   = 1,
        stage1_weight:   float = 0.3,
    ) -> torch.Tensor:
        cands     = candidates[: self.top_k]
        K         = len(cands)
        device    = self.model.device
        digit_ids = self.digit_token_ids.to(device)[:K]
        acc       = torch.zeros(K, device=device)

        for t in range(max(n_tta, 1)):
            if n_tta > 1 and t > 0:
                perm     = list(range(K))
                random.shuffle(perm)
                shuffled = [cands[j] for j in perm]
                inv      = [0] * K
                for orig, new in enumerate(perm):
                    inv[new] = orig
            else:
                shuffled, inv = cands, list(range(K))

            prompt = self._apply_chat_template(
                self.prompt_builder.build(mention_text, mention_caption, shuffled)
            )
            enc  = self.tokenizer(prompt, return_tensors="pt",
                                  truncation=True, max_length=2048).to(device)
            raw  = self.model(**enc).logits[0, -1, :][digit_ids]
            orig = torch.zeros(K, device=device)
            for new_pos, orig_pos in enumerate(inv):
                orig[orig_pos] = raw[new_pos]
            acc = acc + orig

        llm_scores = F.softmax(acc / max(n_tta, 1), dim=-1)

        if stage1_weight > 0:
            s1     = torch.tensor([c.get("score", 0.0) for c in cands],
                                  device=device, dtype=torch.float32)
            lo, hi = s1.min(), s1.max()
            s1_n   = (s1 - lo) / (hi - lo + 1e-8)
            llm_scores = (1 - stage1_weight) * llm_scores + stage1_weight * s1_n

        return llm_scores

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_lora(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"LoRA adapters saved → {path}")

    @classmethod
    def load_lora(cls, base_model_name: str, lora_path: str,
                  top_k: int = 10) -> "LLMReranker":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        inst                = cls.__new__(cls)
        super(LLMReranker, inst).__init__()
        inst.top_k          = top_k
        inst.prompt_builder = PromptBuilder()

        inst.tokenizer = AutoTokenizer.from_pretrained(lora_path, padding_side="left")
        if inst.tokenizer.pad_token is None:
            inst.tokenizer.pad_token = inst.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        inst.model           = PeftModel.from_pretrained(base, lora_path)
        inst.digit_token_ids = inst._get_digit_token_ids(top_k)
        return inst


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class LLMRerankDataset:
    """
    (prompt, label_idx) pairs for training/evaluating the LLM reranker.
    """

    def __init__(
        self,
        stage1_results_path: str,
        captions_path:       str,
        candidate_json_path: str,
        top_k:               int                      = 10,
        prompt_builder:      Optional[PromptBuilder]  = None,
        shuffle_candidates:  bool                     = True,
        mention_meta_path:   Optional[str]            = None,
        entity_aug_path:     Optional[str]            = None,
    ):
        self.top_k              = top_k
        self.builder            = prompt_builder or PromptBuilder()
        self.shuffle_candidates = shuffle_candidates

        # Captions
        self.mention_captions: Dict[str, str] = {}
        self.entity_captions:  Dict[str, str] = {}
        if captions_path and Path(captions_path).exists():
            with open(captions_path) as fh:
                caps = json.load(fh)
            self.mention_captions = caps.get("mentions", {})
            self.entity_captions  = caps.get("entities", {})

        # Text lookup from candidate JSON
        with open(candidate_json_path) as fh:
            raw = json.load(fh)
        self.mention_text: Dict[str, str] = {}
        self.entity_text:  Dict[str, str] = {}
        for rec in raw:
            self.mention_text[rec["mention_id"]] = rec.get("mention_text", "")
            for c in rec.get("candidates", []):
                self.entity_text[c["entity_id"]] = c.get("entity_text", "")

        # Rich mention metadata from EntityExtractionPipeline
        self.mention_meta: Dict[str, Any] = {}
        if mention_meta_path:
            if not Path(mention_meta_path).exists():
                raise FileNotFoundError(
                    f"CRITICAL ERROR: Metadata file not found at: {mention_meta_path}"
                )
            with open(mention_meta_path) as fh:
                meta_list = json.load(fh)

            for item in meta_list:
                mid_num = str(item.get("id", ""))
                if mid_num:
                    self.mention_meta[mid_num] = item

                mid_img = item.get("imgPath", "")
                if mid_img.endswith(".jpg"):
                    mid_img = mid_img.replace(".jpg", "")
                    if mid_img not in self.mention_meta:
                        self.mention_meta[mid_img] = []
                    if isinstance(self.mention_meta[mid_img], list):
                        self.mention_meta[mid_img].append(item)

            logger.info(
                f"SUCCESS: Loaded rich mentions from {Path(mention_meta_path).name}"
            )

        # Entity LLM-augmented summaries
        self.entity_summaries: Dict[str, str] = {}
        for aug_path in [
            entity_aug_path,
            str(Path(stage1_results_path).parent / "augmented_entities.json"),
        ]:
            if aug_path and Path(aug_path).exists():
                with open(aug_path) as fh:
                    raw = json.load(fh)
                if isinstance(raw, list):
                    self.entity_summaries = {str(e["qid"]): e.get("sum", "") for e in raw if e.get("sum")}
                else:
                    self.entity_summaries = raw
                logger.info(f"Loaded {len(self.entity_summaries)} entity summaries")
                break

        # Build samples
        with open(stage1_results_path) as fh:
            results = json.load(fh)
        self.samples = self._build_samples(results)

        n_hard = sum(1 for s in self.samples if s["is_hard"])
        logger.info(
            f"LLMRerankDataset: {len(self.samples)} samples "
            f"({n_hard} hard / {len(self.samples) - n_hard} easy) | "
            f"shuffle={shuffle_candidates} | rich_meta={bool(self.mention_meta)}"
        )

    def _build_samples(self, results: List[Dict]) -> List[Dict]:
        samples, skipped = [], 0
        for r in results:
            mid       = r["mention_id"]
            ranked    = r["ranked"][: self.top_k]
            label_eid = r.get("label_entity_id")

            label_pos = next(
                (i for i, c in enumerate(ranked) if c["entity_id"] == label_eid),
                None,
            )
            if label_pos is None:
                skipped += 1
                continue

            samples.append({
                "mention_id":      mid,
                "mention_text":    self.mention_text.get(mid, ""),
                "mention_caption": self.mention_captions.get(mid, ""),
                "candidates": [
                    {
                        "entity_id":      c["entity_id"],
                        "entity_summary": self.entity_summaries.get(c["entity_id"], ""),
                        "entity_text":    self.entity_text.get(c["entity_id"], ""),
                        "entity_caption": self.entity_captions.get(c["entity_id"], ""),
                        "score":          c["score"],
                    }
                    for c in ranked
                ],
                "label_pos": label_pos,
                "is_hard":   label_pos != 0,
            })

        if skipped:
            logger.warning(
                f"Skipped {skipped} samples where correct entity wasn't in top-{self.top_k}"
            )
        return samples

    def hard_only_view(self) -> "LLMRerankDataset":
        view                    = self.__class__.__new__(self.__class__)
        view.top_k              = self.top_k
        view.builder            = self.builder
        view.shuffle_candidates = self.shuffle_candidates
        view.mention_captions   = self.mention_captions
        view.entity_captions    = self.entity_captions
        view.mention_text       = self.mention_text
        view.entity_text        = self.entity_text
        view.mention_meta       = self.mention_meta
        view.entity_summaries   = self.entity_summaries
        view.samples            = [s for s in self.samples if s["is_hard"]]
        logger.info(f"hard_only_view: {len(view.samples)} hard samples")
        return view

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s          = self.samples[idx]
        candidates = list(s["candidates"])
        label_pos  = s["label_pos"]

        if self.shuffle_candidates:
            perm       = list(range(len(candidates)))
            random.shuffle(perm)
            candidates = [candidates[i] for i in perm]
            label_pos  = perm.index(s["label_pos"])

        meta      = {}
        lookup_id = str(s["mention_id"])

        # 1. Direct lookup
        if lookup_id in self.mention_meta:
            match = self.mention_meta[lookup_id]
            if isinstance(match, dict):
                meta = match
            elif isinstance(match, list):
                for m in match:
                    if m.get("mentions", "") in s["mention_text"]:
                        meta = m
                        break
                if not meta and len(match) > 0:
                    meta = match[0]

        # 2. Fallback by numeric ID
        if not meta:
            for key, val in self.mention_meta.items():
                items = val if isinstance(val, list) else [val]
                for m in items:
                    if str(m.get("id", "")) == lookup_id:
                        meta = m
                        break
                if meta:
                    break

        # 3. Fallback by QID
        if not meta:
            for key, val in self.mention_meta.items():
                items = val if isinstance(val, list) else [val]
                for m in items:
                    if str(m.get("qid", "")) == lookup_id:
                        meta = m
                        break
                if meta:
                    break

        # 4. Text fallback
        if not meta:
            for key, val in self.mention_meta.items():
                items = val if isinstance(val, list) else [val]
                for m in items:
                    m_text     = m.get("mentions", "").strip()
                    m_sent     = m.get("sentence", "").strip()
                    s_text     = s["mention_text"].strip()
                    combo_text = f"{m_text} {m_sent}".strip()
                    if combo_text == s_text or m_text == s_text or m_sent == s_text:
                        meta = m
                        break
                if meta:
                    break

        rich   = _build_rich_mention(s["mention_text"], s["mention_caption"], meta)
        prompt = self.builder.build(
            mention_text=rich["display_text"],
            mention_caption=rich["caption"],
            candidates=candidates,
        )
        return {"mention_id": s["mention_id"], "prompt": prompt, "label_idx": label_pos}

    @staticmethod
    def collate(batch: List[Dict]) -> Dict:
        return {
            "mention_ids": [s["mention_id"] for s in batch],
            "prompts":     [s["prompt"]     for s in batch],
            "label_idx":   torch.tensor(
                [s["label_idx"] for s in batch], dtype=torch.long
            ),
        }
