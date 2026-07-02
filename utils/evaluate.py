"""
evaluate.py — Recall@k, MRR, and delta vs retrieval baseline.

Baseline note
─────────────
The retrieval system pre-sorts candidates by similarity score, so the
correct entity's original retrieval rank is its stored label_idx (0-based).

During training, label_idx is shuffled per sample (position bias fix).
We keep _retrieval_labels as a separate field to preserve the original
retrieval rank for baseline computation regardless of shuffle state.

If _retrieval_labels is not passed, we fall back to label_idx — which
gives the right answer only when validation is NOT shuffled.
"""

from typing import Dict, List, Optional

import torch


def compute_metrics(
    scores:            torch.Tensor,              # [B, N]
    label_idx:         torch.Tensor,              # [B]  shuffled position (correct entity)
    top_k_list:        List[int] = (1, 5, 10),
    retrieval_labels:  Optional[torch.Tensor] = None,  # [B]  original retrieval rank
) -> Dict[str, float]:
    """
    scores         — reranker output scores (higher = better)
    label_idx      — position of correct entity in the (possibly shuffled) candidate list
    retrieval_labels — original rank from retrieval (0 = top-1, 1 = second, ...)
                       used ONLY for baseline_recall computation
                       if None, falls back to label_idx (correct when val not shuffled)
    """
    B, N = scores.shape

    # ── Reranker ranks ────────────────────────────────────────────
    ranked     = scores.argsort(dim=1, descending=True)             # [B, N]
    rerank_pos = (ranked == label_idx.unsqueeze(1)).float() \
                     .argmax(dim=1).float() + 1                      # [B] 1-indexed

    # ── Retrieval baseline ranks ──────────────────────────────────
    # retrieval_pos is the original retrieval rank + 1 (1-indexed)
    # This is independent of any shuffling done during training
    base_labels  = retrieval_labels if retrieval_labels is not None else label_idx
    retrieval_pos = base_labels.float() + 1                         # [B] 1-indexed

    metrics: Dict[str, float] = {}
    for k in top_k_list:
        r_rerank   = (rerank_pos    <= k).float().mean().item()
        r_baseline = (retrieval_pos <= k).float().mean().item()
        metrics[f"recall@{k}"]          = r_rerank
        metrics[f"baseline_recall@{k}"] = r_baseline
        metrics[f"delta@{k}"]           = r_rerank - r_baseline

    metrics["mrr"]          = (1.0 / rerank_pos).mean().item()
    metrics["baseline_mrr"] = (1.0 / retrieval_pos).mean().item()
    metrics["delta_mrr"]    = metrics["mrr"] - metrics["baseline_mrr"]

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}
