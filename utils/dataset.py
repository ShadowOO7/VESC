"""
dataset.py — CandidateDataset, collate_fn, build_dataloaders

Expected JSON schema (per element in the list):
{
  "mention_id":    "m001",
  "mention_text":  "Steve Jobs unveiling the first iPhone",
  "mention_image": "mentions/m001.jpg",   // relative to image_dir; omit if absent
  "label_idx":     4,                     // index of correct candidate (0-based)
  "candidates": [
    {
      "entity_id":    "Q19837",
      "entity_text":  "Steve Jobs American entrepreneur and Apple co-founder",
      "entity_image": "entities/Q19837.jpg",  // optional
      "is_correct":   true    // alternative to label_idx at the mention level
    },
    ...  // exactly 50 entries (or will be trimmed / padded)
  ]
}
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class CandidateDataset(Dataset):
    """
    One sample  = one mention + its N entity candidates.

    Tensors returned per sample
    ───────────────────────────
    m_input_ids        [T]
    m_attention_mask   [T]
    m_pixel_values     [3, 224, 224]  — zeros when no image

    e_input_ids        [N, T]
    e_attention_mask   [N, T]
    e_pixel_values     [N, 3, 224, 224]

    label_idx          scalar long (training only)
    """

    def __init__(
        self,
        data_path: str,
        processor: CLIPProcessor,
        image_dir: str = "",
        num_candidates: int = 50,
        max_text_length: int = 77,
        is_train: bool = True,
    ):
        self.processor       = processor
        self.image_dir       = Path(image_dir) if image_dir else Path(".")
        self.num_candidates  = num_candidates
        self.max_text_length = max_text_length
        self.is_train        = is_train

        with open(data_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        self.data = self._prepare(raw)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare(self, raw: List[Dict]) -> List[Dict]:
        out, skipped = [], 0

        for sample in raw:
            cands = list(sample["candidates"])

            # Trim / right-pad to exactly num_candidates
            if len(cands) > self.num_candidates:
                cands = cands[: self.num_candidates]
            elif len(cands) < self.num_candidates:
                cands += [cands[-1]] * (self.num_candidates - len(cands))
            sample["candidates"] = cands

            # Resolve label
            if self.is_train:
                if "label_idx" not in sample:
                    for i, c in enumerate(cands):
                        if c.get("is_correct", False):
                            sample["label_idx"] = i
                            break
                    else:
                        skipped += 1
                        continue

                lbl = sample["label_idx"]
                if not (0 <= lbl < self.num_candidates):
                    skipped += 1
                    continue

            out.append(sample)

        if skipped:
            logger.warning(f"Skipped {skipped} samples (missing or out-of-range label)")

        return out

    def _load_image(self, rel_path: str) -> Optional[Image.Image]:
        if not rel_path:
            return None
        p = Path(rel_path)
        full = p if p.is_absolute() else self.image_dir / p
        try:
            return Image.open(full).convert("RGB")
        except Exception:
            return None

    def _tok(self, text: str):
        enc = self.processor(
            text=text or "[UNK]",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def _pix(self, image: Optional[Image.Image]) -> torch.Tensor:
        if image is None:
            return torch.zeros(3, 224, 224)
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.data[idx]

        # Mention
        m_ids, m_mask = self._tok(s["mention_text"])
        m_pix         = self._pix(self._load_image(s.get("mention_image", "")))

        # Candidates
        e_ids_l, e_mask_l, e_pix_l = [], [], []
        for c in s["candidates"]:
            ids, mask = self._tok(c["entity_text"])
            pix       = self._pix(self._load_image(c.get("entity_image", "")))
            e_ids_l.append(ids)
            e_mask_l.append(mask)
            e_pix_l.append(pix)

        item: Dict[str, Any] = {
            "mention_id":       s["mention_id"],
            "entity_ids":       [c["entity_id"] for c in s["candidates"]],
            # Mention tensors
            "m_input_ids":      m_ids,
            "m_attention_mask": m_mask,
            "m_pixel_values":   m_pix,
            # Entity tensors  [N, ...]
            "e_input_ids":      torch.stack(e_ids_l),
            "e_attention_mask": torch.stack(e_mask_l),
            "e_pixel_values":   torch.stack(e_pix_l),
        }

        if self.is_train:
            item["label_idx"] = torch.tensor(s["label_idx"], dtype=torch.long)

        return item


# ─────────────────────────────────────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Input : list of B per-sample dicts
    Output: batched dict

      m_input_ids        [B, T]
      m_attention_mask   [B, T]
      m_pixel_values     [B, 3, H, W]
      e_input_ids        [B, N, T]
      e_attention_mask   [B, N, T]
      e_pixel_values     [B, N, 3, H, W]
      label_idx          [B]   (if present)
    """
    TENSOR_KEYS = [
        "m_input_ids", "m_attention_mask", "m_pixel_values",
        "e_input_ids", "e_attention_mask", "e_pixel_values",
    ]

    out: Dict[str, Any] = {
        "mention_ids": [s["mention_id"] for s in batch],
        "entity_ids":  [s["entity_ids"] for s in batch],
    }

    for k in TENSOR_KEYS:
        out[k] = torch.stack([s[k] for s in batch])

    if "label_idx" in batch[0]:
        out["label_idx"] = torch.stack([s["label_idx"] for s in batch])

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(train_cfg, model_cfg, processor):
    """Return (train_loader, val_loader)."""

    def _make(path, is_train, shuffle):
        ds = CandidateDataset(
            path, processor,
            image_dir=train_cfg.image_dir,
            num_candidates=train_cfg.num_candidates,
            max_text_length=train_cfg.max_text_length,
            is_train=is_train,
        )
        return DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            persistent_workers=(train_cfg.num_workers > 0),
        )

    return (
        _make(train_cfg.train_path, is_train=True,  shuffle=True),
        _make(train_cfg.val_path,   is_train=True,  shuffle=False),
    )
