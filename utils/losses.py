"""
losses.py — Reranking loss for 50-candidate entity linking

Four terms
──────────
1. Main CE (listwise)
   Cross entropy over N=50 candidate scores.
   Pushes the correct candidate to rank 1.

2. Auxiliary sub-losses (weighted by aux_weight)
   Same CE applied separately to global_scores and local_scores.
   Provides independent gradient signal to each head.

3. Triplet margin loss (weighted by margin_weight)
   Penalises cases where the hardest negative is within `margin`
   of the correct candidate.  Encourages a safety gap.

4. Recall@k preservation loss (weighted by recall_k_weight)  ← NEW
   Directly penalises the model whenever the correct candidate
   falls outside the top-k positions.  Without this, CE only
   cares about rank-1, and the model is free to push the correct
   answer to rank 25 on uncertain samples — destroying recall@10.

   Implementation: for each sample, find the (k)th-hardest negative
   (i.e. the negative with the k-th highest score among all negatives).
   Apply a margin hinge against the correct score.  If the correct
   candidate beats the k-th hardest negative by at least `margin`,
   the loss is zero — recall@k is preserved.  Otherwise the model
   is penalised proportionally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class RerankingLoss(nn.Module):

    def __init__(
        self,
        temperature:     float = 1.0,
        aux_weight:      float = 0.10,
        margin_weight:   float = 0.05,
        margin:          float = 0.20,
        recall_k:        int   = 10,
        recall_k_weight: float = 0.30,
    ):
        super().__init__()
        self.temperature     = temperature
        self.aux_weight      = aux_weight
        self.margin_weight   = margin_weight
        self.margin          = margin
        self.recall_k        = recall_k
        self.recall_k_weight = recall_k_weight
        self.ce              = nn.CrossEntropyLoss()

    def forward(
        self,
        scores:        torch.Tensor,             # [B, N]
        label_idx:     torch.Tensor,             # [B]
        global_scores: Optional[torch.Tensor] = None,   # [B, N]
        local_scores:  Optional[torch.Tensor] = None,   # [B, N]
    ) -> Dict[str, torch.Tensor]:
        B, N = scores.shape
        device = scores.device
        arange = torch.arange(B, device=device)

        # ── 1. Main listwise CE ───────────────────────────────────
        main_loss = self.ce(scores / self.temperature, label_idx)

        # ── 2. Auxiliary sub-losses ───────────────────────────────
        aux_loss = scores.new_tensor(0.0)
        if global_scores is not None:
            aux_loss = aux_loss + self.ce(global_scores / self.temperature, label_idx)
        if local_scores is not None:
            aux_loss = aux_loss + self.ce(local_scores  / self.temperature, label_idx)

        # ── 3. Triplet margin (hardest negative) ──────────────────
        correct = scores[arange, label_idx].unsqueeze(1)          # [B, 1]

        neg_mask               = torch.ones_like(scores, dtype=torch.bool)
        neg_mask[arange, label_idx] = False
        neg_scores             = scores.masked_fill(~neg_mask, float("-inf"))

        hardest_neg            = neg_scores.max(dim=1).values      # [B]
        margin_loss            = F.relu(
            hardest_neg - correct.squeeze(1) + self.margin
        ).mean()

        # ── 4. Recall@k preservation ──────────────────────────────
        # Among all negatives, find the k-th highest score.
        # The correct candidate must beat this score by at least `margin`
        # for it to be in the top-k.  Hinge loss if it doesn't.
        k = min(self.recall_k, N - 1)   # can't exceed number of negatives

        # Replace correct with -inf so it can't be selected as its own negative
        neg_for_recall         = scores.masked_fill(~neg_mask, float("-inf"))  # [B, N]

        # top-k scores among negatives; we care about the k-th (the boundary)
        topk_neg_scores        = torch.topk(neg_for_recall, k, dim=1).values   # [B, k]
        kth_neg_score          = topk_neg_scores[:, -1]                        # [B]

        # If correct_score > kth_neg_score + margin → no loss (correct is safely in top-k)
        # If correct_score < kth_neg_score          → penalised
        recall_k_loss          = F.relu(
            kth_neg_score - correct.squeeze(1) + self.margin
        ).mean()

        # ── Total ─────────────────────────────────────────────────
        total = (
            main_loss
            + self.aux_weight      * aux_loss
            + self.margin_weight   * margin_loss
            + self.recall_k_weight * recall_k_loss
        )

        return {
            "loss":           total,
            "main_loss":      main_loss.detach(),
            "aux_loss":       aux_loss.detach(),
            "margin_loss":    margin_loss.detach(),
            "recall_k_loss":  recall_k_loss.detach(),
        }
