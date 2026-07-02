"""
cached_dataset.py — CachedCandidateDataset backed by HDF5.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CachedCandidateDataset(Dataset):

    def __init__(
        self,
        data_path:          str,
        cache_dir:          str,
        num_candidates:     int  = 50,
        is_train:           bool = True,
        shuffle_candidates: bool = False,   # caller sets this explicitly
    ):
        self.cache_dir          = Path(cache_dir)
        self.h5_path            = self.cache_dir / "clip_cache.h5"
        self.num_candidates     = num_candidates
        self.is_train           = is_train
        self.shuffle_candidates = shuffle_candidates
        self._h5_handle: Optional[h5py.File] = None

        # ── Sanity check HDF5 ─────────────────────────────────────
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 cache not found: {self.h5_path}")
        try:
            with h5py.File(self.h5_path, "r") as f:
                n_ments  = f["mentions"]["text_cls"].shape[0]
                n_ents   = f["entities"]["text_cls"].shape[0]
                text_dim = f["mentions"]["text_cls"].shape[1]
                img_seq  = f["entities"]["image_tokens"].shape[1]
                img_dim  = f["entities"]["image_tokens"].shape[-1]
            logger.info(
                f"HDF5 OK — {n_ments} mentions, {n_ents} entities | "
                f"text_dim={text_dim}, img_tokens={img_seq}×{img_dim}"
            )
        except Exception as e:
            raise RuntimeError(f"Cannot open HDF5 cache: {e}")

        # ── Load index ─────────────────────────────────────────────
        idx_path = self.cache_dir / "cache_index.json"
        with open(idx_path, "r") as fh:
            index = json.load(fh)
        self.mention_index: Dict[str, int] = index["mentions"]
        self.entity_index:  Dict[str, int] = index["entities"]

        # ── Load candidate records ─────────────────────────────────
        with open(data_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.data = self._prepare(raw)

        mode = "TRAIN shuffle=ON" if shuffle_candidates else "VAL  shuffle=OFF"
        logger.info(
            f"[{mode}] CachedCandidateDataset: "
            f"{len(self.data)} samples from {Path(data_path).name}"
        )

    # ── Lazy h5 handle — plain "r", no swmr ───────────────────────────────

    @property
    def _h5(self) -> h5py.File:
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_path, "r")
        return self._h5_handle

    # ── Prepare ────────────────────────────────────────────────────────────

    def _prepare(self, raw: List[Dict]) -> List[Dict]:
        out, skipped = [], 0
        for sample in raw:
            cands = list(sample["candidates"])
            if len(cands) > self.num_candidates:
                cands = cands[: self.num_candidates]
            elif len(cands) < self.num_candidates:
                cands += [cands[-1]] * (self.num_candidates - len(cands))
            sample["candidates"] = cands

            # Always resolve labels — needed for eval regardless of train/val split
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

            mid = sample["mention_id"]
            if mid not in self.mention_index:
                skipped += 1
                continue

            entity_rows, missing = [], False
            for c in cands:
                eid = c["entity_id"]
                if eid not in self.entity_index:
                    missing = True
                    break
                entity_rows.append(self.entity_index[eid])
            if missing:
                skipped += 1
                continue

            sample["_mention_row"] = self.mention_index[mid]
            sample["_entity_rows"] = entity_rows
            out.append(sample)

        if skipped:
            logger.warning(f"Skipped {skipped} samples")
        return out

    # ── HDF5 reads ─────────────────────────────────────────────────────────

    def _read_mention(self, row: int) -> Dict[str, torch.Tensor]:
        return {
            "text_cls":    torch.from_numpy(
                self._h5["mentions"]["text_cls"][row].astype(np.float32)
            ),
            "text_tokens": torch.from_numpy(
                self._h5["mentions"]["text_tokens"][row].astype(np.float32)
            ),
        }

    def _read_entities_batched(self, rows: List[int]) -> Dict[str, torch.Tensor]:
        """3 bulk reads (sorted indices) instead of 3×N individual reads."""
        rows_arr = np.array(rows, dtype=np.int64)
        sort_idx = np.argsort(rows_arr)
        sorted_r = rows_arr[sort_idx].tolist()
        unsort   = np.argsort(sort_idx)

        eg    = self._h5["entities"]
        t_cls = eg["text_cls"]    [sorted_r]
        t_tok = eg["text_tokens"] [sorted_r]
        i_tok = eg["image_tokens"][sorted_r]

        return {
            "text_cls":     torch.from_numpy(t_cls[unsort].astype(np.float32)),
            "text_tokens":  torch.from_numpy(t_tok[unsort].astype(np.float32)),
            "image_tokens": torch.from_numpy(i_tok[unsort].astype(np.float32)),
        }

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s           = self.data[idx]
        entity_rows = list(s["_entity_rows"])
        entity_ids  = [c["entity_id"] for c in s["candidates"]]
        label_idx   = s.get("label_idx", 0)

        # ── Shuffle only for TRAINING ──────────────────────────────
        if self.shuffle_candidates:
            perm        = list(range(len(entity_rows)))
            random.shuffle(perm)
            entity_rows = [entity_rows[i] for i in perm]
            entity_ids  = [entity_ids[i]  for i in perm]
            label_idx   = perm.index(label_idx)

        m = self._read_mention(s["_mention_row"])
        e = self._read_entities_batched(entity_rows)

        item: Dict[str, Any] = {
            "mention_id":     s["mention_id"],
            "entity_ids":     entity_ids,
            "m_text_cls":     m["text_cls"],
            "m_text_tokens":  m["text_tokens"],
            "e_text_cls":     e["text_cls"],
            "e_text_tokens":  e["text_tokens"],
            "e_image_tokens": e["image_tokens"],
            # Store original retrieval rank separately for baseline metric
            "_retrieval_label": torch.tensor(
                s.get("label_idx", 0), dtype=torch.long
            ),
        }
        # Always include label_idx — eval_epoch needs it for val too
        item["label_idx"] = torch.tensor(label_idx, dtype=torch.long)
        return item


# ─────────────────────────────────────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────────────────────────────────────

CACHED_TENSOR_KEYS = [
    "m_text_cls", "m_text_tokens",
    "e_text_cls", "e_text_tokens", "e_image_tokens",
]


def cached_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mention_ids":       [s["mention_id"] for s in batch],
        "entity_ids":        [s["entity_ids"] for s in batch],
        "_retrieval_labels": torch.stack([s["_retrieval_label"] for s in batch]),
    }
    for k in CACHED_TENSOR_KEYS:
        out[k] = torch.stack([s[k] for s in batch])
    if "label_idx" in batch[0]:
        out["label_idx"] = torch.stack([s["label_idx"] for s in batch])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_cached_dataloaders(train_cfg, cache_dir: str):

    def _make(path, is_train):
        ds = CachedCandidateDataset(
            path,
            cache_dir=cache_dir,
            num_candidates=train_cfg.num_candidates,
            is_train=is_train,
            # ─────────────────────────────────────────────────────
            # TRAIN: shuffle=True  eliminates position bias
            # VAL:   shuffle=False preserves retrieval order so
            #        baseline_recall metrics are meaningful
            # ─────────────────────────────────────────────────────
            shuffle_candidates=is_train,
        )
        return DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=is_train,           # shuffle sample order for train
            collate_fn=cached_collate_fn,
            num_workers=train_cfg.num_workers,
            prefetch_factor=2 if train_cfg.num_workers > 0 else None,
            pin_memory=True,
            # ─────────────────────────────────────────────────────
            # persistent_workers=False prevents h5py chunk-cache
            # memory leak that causes epochs to grow from 45min→9h
            # ─────────────────────────────────────────────────────
            persistent_workers=False,
        )

    return (
        _make(train_cfg.train_path, is_train=True),
        _make(train_cfg.val_path,   is_train=False),
    )
