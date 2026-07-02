"""
llm_trainer.py — Training loop for Stage 3 LLM reranker.

Changes vs previous version
────────────────────────────
1. Fixed deprecated torch.cuda.amp imports → torch.amp
2. Curriculum training:
     Epoch 1  — hard samples only  (~11% of data where Stage 1 was wrong)
     Epochs 2+ — full shuffled dataset
   Forces the model to learn genuine reasoning before seeing the easy
   majority that would otherwise dominate every gradient step.
3. _eval_recall1 → _eval_metrics: now logs R@1, R@5, R@10 so you can
   see if the model is improving on hard cases across epochs.
4. LR reduced to 5e-5 (set in train_pipeline.py _LLMCfg.llm_lr).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast          # ← fixed (was torch.cuda.amp)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import aggregate_metrics
from llm_reranker_byme import LLMRerankDataset, LLMReranker, PromptBuilder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM STAGE TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class LLMStageTrainer:

    def __init__(
        self,
        llm_reranker:   LLMReranker,
        train_dataset:  LLMRerankDataset,
        val_dataset:    LLMRerankDataset,
        cfg,
    ):
        self.model       = llm_reranker
        self.train_ds    = train_dataset
        self.val_ds      = val_dataset
        self.cfg         = cfg
        self.output_dir  = Path(cfg.output_dir) / "llm_stage"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_params  = sum(p.numel() for p in trainable)
        logger.info(f"Training {n_params / 1e6:.1f}M LoRA parameters")

        self.optimizer = AdamW(
            trainable,
            lr=getattr(cfg, "llm_lr", 5e-5),        # ← lowered default to 5e-5
            weight_decay=1e-2,
        )

        self.scaler      = GradScaler("cuda", enabled=cfg.fp16)   # ← fixed API
        self.best_recall = 0.0
        self.global_step = 0

    # ── DataLoader helper ─────────────────────────────────────────────────────

    def _make_loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=getattr(self.cfg, "llm_batch_size", 2),
            shuffle=shuffle,
            collate_fn=LLMRerankDataset.collate,
            num_workers=0,    # tokenizer not fork-safe
        )

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        n_epochs = getattr(self.cfg, "llm_epochs", 3)

        # ── Hard sample set (Stage 1 wrong) ──────────────────────────────────
        hard_ds    = self.train_ds.hard_only_view()
        n_hard     = len(hard_ds)
        n_total    = len(self.train_ds)
        logger.info(
            f"Curriculum training: {n_hard} hard samples, "
            f"{n_total} total | {n_epochs} epochs"
        )

        for epoch in range(n_epochs):

            # ── Epoch 0: hard samples only ────────────────────────────────────
            # Forces the model to learn genuine reasoning on cases where
            # Stage 1 failed before the easy majority dominates gradients.
            if epoch == 0:
                loader = self._make_loader(hard_ds, shuffle=True)
                logger.info(
                    f"Epoch 1/{n_epochs} — HARD SAMPLES ONLY "
                    f"({n_hard} samples, curriculum warmup)"
                )
            else:
                loader = self._make_loader(self.train_ds, shuffle=True)
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} — FULL DATASET "
                    f"({n_total} samples, shuffled candidates)"
                )

            self.model.train()
            epoch_loss  = 0.0
            n_steps     = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", dynamic_ncols=True)

            for batch in pbar:
                self.optimizer.zero_grad()

                with autocast("cuda", enabled=self.cfg.fp16):   # ← fixed API
                    out = self.model(
                        prompts=batch["prompts"],
                        label_idx=batch["label_idx"],
                    )

                self.scaler.scale(out["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss   += out["loss"].item()
                n_steps      += 1
                self.global_step += 1

                pbar.set_postfix(loss=f"{epoch_loss / n_steps:.4f}",
                                 step=self.global_step)

            avg_loss = epoch_loss / max(n_steps, 1)

            # ── Eval after every epoch ────────────────────────────────────────
            metrics = self._eval_metrics()
            recall1 = metrics["recall@1"]

            logger.info(
                f"Epoch {epoch+1}/{n_epochs} | loss={avg_loss:.4f} | "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

            if recall1 > self.best_recall:
                self.best_recall = recall1
                self.model.save_lora(str(self.output_dir / "lora_best"))
                logger.info(f"  ↑ New best Recall@1: {recall1:.4f}")

        logger.info(f"Training complete. Best Recall@1: {self.best_recall:.4f}")

    # ── Eval ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_metrics(self) -> Dict[str, float]:
        """
        Evaluate on the full (unshuffled) validation set.
        Returns Recall@1, Recall@5, Recall@10 and MRR.
        Val dataset uses shuffle_candidates=False so positions are stable.
        """
        self.model.eval()
        val_loader = self._make_loader(self.val_ds, shuffle=False)

        top_k      = getattr(self.cfg, "llm_top_k", 10)
        recalls    = {k: 0 for k in [1, 5, 10] if k <= top_k}
        rr_sum     = 0.0
        total      = 0

        for batch in tqdm(val_loader, desc="  [Val]", leave=False, dynamic_ncols=True):
            out    = self.model(prompts=batch["prompts"], label_idx=batch["label_idx"])
            # digit_logits: [B, K] — higher = model prefers that candidate
            ranked = out["digit_logits"].argsort(dim=-1, descending=True)   # [B, K]
            labels = batch["label_idx"].to(out["digit_logits"].device)       # [B]

            for i in range(labels.shape[0]):
                correct = labels[i].item()
                # 1-indexed rank of correct candidate in model's ranking
                pos    = (ranked[i] == correct).nonzero(as_tuple=True)[0]
                rank   = pos[0].item() + 1 if len(pos) > 0 else top_k + 1

                for k in recalls:
                    if rank <= k:
                        recalls[k] += 1
                rr_sum += 1.0 / rank
                total  += 1

        self.model.train()
        metrics = {f"recall@{k}": recalls[k] / max(total, 1) for k in recalls}
        metrics["mrr"] = rr_sum / max(total, 1)
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 TOP-K RESULT GENERATOR  (used by train_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_stage1_results(
    stage1_model,
    data_loader,
    device: torch.device,
    fp16:   bool = True,
    top_k:  int  = 10,
) -> List[Dict]:
    """
    Run Stage 1 model over a DataLoader and return top-K results per mention.
    Output format: [{"mention_id", "label_entity_id", "ranked": [{"entity_id", "score"}]}]
    """
    stage1_model.eval()
    results = []

    for batch in data_loader:
        batch_d = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with autocast("cuda", enabled=fp16):                # ← fixed API
            scores, _, _ = stage1_model(batch_d)            # [B, N]

        B, N       = scores.shape
        scores_cpu = scores.cpu()

        for i in range(B):
            mid   = batch["mention_ids"][i]
            eids  = batch["entity_ids"][i]
            label = batch["label_idx"][i].item() if "label_idx" in batch else None
            s     = scores_cpu[i]

            order  = s.argsort(descending=True).tolist()[: top_k]
            ranked = [{"entity_id": eids[j], "score": float(s[j])} for j in order]

            results.append({
                "mention_id":      mid,
                "label_entity_id": eids[label] if label is not None else None,
                "ranked":          ranked,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TWO-STAGE INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class TwoStagePipeline:

    def __init__(
        self,
        stage1_model,
        llm_reranker:         LLMReranker,
        captions_path:        str,
        candidate_json_path:  str,
        top_k:                int = 10,
        device:               str = "cuda",
        fp16:                 bool = True,
    ):
        self.stage1  = stage1_model
        self.llm     = llm_reranker
        self.top_k   = top_k
        self.device  = torch.device(device)
        self.fp16    = fp16
        self.builder = PromptBuilder()

        with open(captions_path, "r") as fh:
            caps = json.load(fh)
        self.mention_captions: Dict[str, str] = caps.get("mentions", {})
        self.entity_captions:  Dict[str, str] = caps.get("entities", {})

        with open(candidate_json_path, "r") as fh:
            raw = json.load(fh)
        self.mention_text: Dict[str, str] = {}
        self.entity_text:  Dict[str, str] = {}
        for rec in raw:
            self.mention_text[rec["mention_id"]] = rec.get("mention_text", "")
            for c in rec.get("candidates", []):
                self.entity_text[c["entity_id"]] = c.get("entity_text", "")

    def rerank_batch(self, stage1_batch: Dict, stage1_scores: torch.Tensor) -> List[Dict]:
        B       = stage1_scores.shape[0]
        results = []

        for i in range(B):
            mid   = stage1_batch["mention_ids"][i]
            eids  = stage1_batch["entity_ids"][i]
            s     = stage1_scores[i].cpu()

            order      = s.argsort(descending=True).tolist()[: self.top_k]
            stage1_top = [
                {
                    "entity_id":      eids[j],
                    "entity_text":    self.entity_text.get(eids[j], ""),
                    "entity_caption": self.entity_captions.get(eids[j], ""),
                    "score":          float(s[j]),
                }
                for j in order
            ]

            llm_scores   = self.llm.score_candidates(
                mention_text=self.mention_text.get(mid, ""),
                mention_caption=self.mention_captions.get(mid, ""),
                candidates=stage1_top,
            ).cpu()
            llm_order    = llm_scores.argsort(descending=True).tolist()
            final_ranked = [stage1_top[j]["entity_id"] for j in llm_order]

            results.append({
                "mention_id":      mid,
                "top1":            final_ranked[0],
                "ranked_entities": final_ranked,
                "llm_scores":      llm_scores[llm_order].tolist(),
                "stage1_scores":   [stage1_top[j]["score"] for j in llm_order],
            })

        return results

    @torch.no_grad()
    def rerank_file(
        self,
        candidate_path: str,
        output_path:    Optional[str] = None,
        data_loader=None,
    ) -> List[Dict]:
        self.stage1.eval()
        self.llm.eval()
        all_results = []

        for batch in data_loader:
            batch_d = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with autocast("cuda", enabled=self.fp16):       # ← fixed API
                scores, _, _ = self.stage1(batch_d)

            all_results.extend(self.rerank_batch(batch, scores))

        logger.info(f"Reranked {len(all_results)} mentions (two-stage)")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(all_results, fh, indent=2, ensure_ascii=False)
            logger.info(f"Saved → {output_path}")

        return all_results

    def evaluate(self, results: List[Dict], ground_truth: Dict[str, str]) -> Dict:
        recalls = {1: 0, 5: 0, 10: 0}
        rr_sum  = 0.0
        total   = len(results)

        for r in results:
            mid     = r["mention_id"]
            correct = ground_truth.get(mid)
            ranked  = r["ranked_entities"]
            for k in recalls:
                if correct in ranked[:k]:
                    recalls[k] += 1
            if correct in ranked:
                rr_sum += 1.0 / (ranked.index(correct) + 1)

        return {f"recall@{k}": v / total for k, v in recalls.items()} | \
               {"mrr": rr_sum / total}
