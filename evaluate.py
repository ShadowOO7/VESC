"""
evaluate_test.py — Full pipeline evaluation on new test data.

Pipeline
────────
  test.json (50 candidates per mention)
       │
       ▼
  Stage 1: CrossEncoderReranker (BAAI/bge-reranker-v2-m3, fine-tuned)
           scores all 50 candidates, returns top-10
       │
       ▼
  Stage 3: Qwen LLM reranker (batched)
           re-ranks top-10 using listwise reasoning
       │
       ▼
  Metrics: Recall@1/5/10, MRR, per-sample results JSON

Usage
─────
  # Full evaluation (Stage 1 + Stage 3)
  python evaluate_test.py \
    --test        /path/to/wikidiverse_top50_ready_test.json \
    --stage1-ckpt /home/kmpooja/ram_sandbox/results/cross_encoder/best/ \
    --lora-path   /home/kmpooja/ram_sandbox/results/llm_stage/lora_best \
    --llm-model   /home/kmpooja/UniMEL/models/Qwen2-7B-Instruct \
    --captions    /home/kmpooja/ram_sandbox/src/cross/data/captions.json \
    --output      results/test_results.json

  # Stage 1 only (no LLM — fast sanity check)
  python evaluate_test.py \
    --test        /path/to/wikidiverse_top50_ready_test.json \
    --stage1-ckpt /home/kmpooja/ram_sandbox/results/cross_encoder/best/ \
    --output      results/stage1_only.json \
    --stage1-only

  # No ground truth labels (inference only)
  python evaluate_test.py \\
    --test        /path/to/test_no_labels.json \\
    --stage1-ckpt /home/kmpooja/ram_sandbox/results/cross_encoder/best/ \\
    --lora-path   /home/kmpooja/ram_sandbox/results/llm_stage/lora_best \\
    --llm-model   /home/kmpooja/UniMEL/models/Qwen2-7B-Instruct \\
    --output      results/test_results.json \\
    --no-metrics

NOTE: The --stage1-ckpt must be the cross-encoder directory
      (results/cross_encoder/best/), NOT a .pt file.
      The old CachedEntityReranker is no longer used.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: CrossEncoderReranker → top-K per mention
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(
    test_json:    str,
    stage1_ckpt:  str,
    top_k:        int = 10,
    max_length:   int = 128,
) -> List[Dict]:
    """
    Run the fine-tuned CrossEncoderReranker over the test set.

    Reads text directly from the JSON — no HDF5 cache needed.
    Returns list of dicts in STAGE1_TOPK format consumed by Stage 3.

    stage1_ckpt must be the cross-encoder directory produced by
    CrossEncoderReranker.save() — i.e. results/cross_encoder/best/.
    """
    from cross_encoder import CrossEncoderReranker

    logger.info("=== Stage 1: CrossEncoder scoring 50 candidates per mention ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if stage1_ckpt and Path(stage1_ckpt).exists():
        logger.info(f"Loading fine-tuned cross-encoder from {stage1_ckpt}")
        reranker            = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.fp16       = True
        reranker.max_length = max_length
        reranker.device     = torch.device(device)
        reranker.load(stage1_ckpt)
    else:
        logger.warning(
            f"Cross-encoder checkpoint not found at {stage1_ckpt!r}. "
            "Falling back to zero-shot base model."
        )
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            device=device,
            fp16=True,
            max_length=max_length,
        )

    # rerank_json handles label resolution and outputs STAGE1_TOPK format
    results = reranker.rerank_json(
        json_path=test_json,
        output_path=None,   # don't save intermediate file
        top_k=top_k,
    )

    logger.info(f"Stage 1 complete — {len(results)} mentions, top-{top_k} each")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: Qwen LLM reranker over Stage 1 top-K
# ─────────────────────────────────────────────────────────────────────────────

def run_stage3(
    stage1_results: List[Dict],
    test_json:      str,
    captions_json:  str,
    lora_path:      str,
    llm_model:      str,
    meta_path:      str,
    top_k:          int = 10,
    batch_size:     int = 8,
    entity_aug_path: Optional[str] = None,
) -> List[Dict]:
    """
    Run Stage 3 Qwen LLM reranker over Stage 1 top-K results.

    Uses shuffle_candidates=False (stable order for eval — we are not
    training here, so position bias doesn't matter).
    """
    from llm_reranker_byme import LLMRerankDataset, LLMReranker

    logger.info("=== Stage 3: Qwen LLM reranking top-K candidates ===")

    # Write stage1 results to a temp file — LLMRerankDataset reads from JSON
    tmp_path = Path("/tmp/_eval_stage1_topk.json")
    with open(tmp_path, "w") as f:
        json.dump(stage1_results, f)

    if lora_path:
        llm = LLMReranker.load_lora(
            base_model_name=llm_model,
            lora_path=lora_path,
            top_k=top_k,
        )
    else:
        llm = LLMReranker.load_base(
            base_model_name=llm_model,
            top_k=top_k,
        )

    ds = LLMRerankDataset(
        stage1_results_path=str(tmp_path),
        captions_path=captions_json,
        candidate_json_path=test_json,
        top_k=top_k,
        shuffle_candidates=False,   # stable order for eval
        mention_meta_path=meta_path,
        entity_aug_path=entity_aug_path,
    )

    logger.info(
        f"LLM dataset: {len(ds)} samples "
        f"({len(stage1_results) - len(ds)} skipped — correct entity not in top-{top_k})"
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LLMRerankDataset.collate,
        num_workers=0,   # tokenizer not fork-safe
    )

    # Build lookup from mention_id → Stage 1 entry for score fusion
    s1_lookup = {r["mention_id"]: r for r in stage1_results}

    final_results = []
    llm.model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Stage 3 LLM", dynamic_ncols=True):
            out    = llm(
                prompts=batch["prompts"],
                label_idx=batch["label_idx"],
            )
            # digit_logits [B, K]: logit for each candidate position 1..K
            ranked = out["digit_logits"].argsort(dim=-1, descending=True)  # [B, K]

            for i, mid in enumerate(batch["mention_ids"]):
                s1    = s1_lookup[mid]
                cands = s1["ranked"]   # Stage 1 ordered candidates
                order = ranked[i].cpu().tolist()

                final_results.append({
                    "mention_id":        mid,
                    "label_entity_id":   s1.get("label_entity_id"),
                    "ranked_entity_ids": [cands[j]["entity_id"]
                                          for j in order if j < len(cands)],
                    "llm_scores":        out["digit_logits"][i].softmax(dim=-1)
                                             .cpu().tolist(),
                    "stage1_scores":     [cands[j]["score"]
                                          for j in order if j < len(cands)],
                })

    logger.info(f"Stage 3 complete — {len(final_results)} mentions reranked")
    return final_results


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    results:    List[Dict],
    use_stage1: bool = False,
) -> Dict[str, float]:
    """
    Recall@1/5/10 and MRR.

    use_stage1=True  → evaluate Stage 1 top-K order  (field "ranked")
    use_stage1=False → evaluate LLM-reranked order    (field "ranked_entity_ids")
    """
    recalls = {1: 0, 5: 0, 10: 0}
    rr_sum  = 0.0
    total   = 0
    skipped = 0

    for r in results:
        label_eid = r.get("label_entity_id")
        if label_eid is None:
            skipped += 1
            continue

        ranked = (
            [c["entity_id"] for c in r["ranked"]]
            if use_stage1
            else r["ranked_entity_ids"]
        )

        if label_eid in ranked:
            rank    = ranked.index(label_eid) + 1
            rr_sum += 1.0 / rank
            for k in recalls:
                if rank <= k:
                    recalls[k] += 1
        total += 1

    if skipped:
        logger.warning(f"Skipped {skipped} samples with no label_entity_id")

    if total == 0:
        return {}

    metrics = {f"recall@{k}": recalls[k] / total for k in sorted(recalls)}
    metrics["mrr"]   = rr_sum / total
    metrics["total"] = total
    return metrics


def print_comparison(s1: Dict, combined: Dict):
    logger.info(f"\n{'Metric':15s}  {'Stage 1':>10}  {'Stage 1+3':>10}  {'Delta':>9}")
    logger.info("-" * 52)
    for k in ["recall@1", "recall@5", "recall@10", "mrr"]:
        v1  = s1.get(k, 0.0)
        v12 = combined.get(k, 0.0)
        logger.info(f"{k:15s}  {v1:>10.4f}  {v12:>10.4f}  {v12-v1:>+9.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline evaluation: CrossEncoder Stage 1 + Qwen Stage 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--test",         required=True,
                        help="Test JSON with 50 candidates per mention")
    parser.add_argument("--stage1-ckpt",
                        default="/home/kmpooja/ram_sandbox/results/cross_encoder/best/",
                        dest="stage1_ckpt",
                        help="Fine-tuned cross-encoder directory "
                             "(results/cross_encoder/best/ — NOT a .pt file)")
    parser.add_argument("--lora-path",    default=None, dest="lora_path",
                        help="Stage 3 LoRA adapters directory")
    parser.add_argument("--llm-model",
                        default="/home/kmpooja/UniMEL/models/Qwen2-7B-Instruct",
                        dest="llm_model",
                        help="Base LLM model name or local path")
    parser.add_argument("--captions",
                        default="/home/kmpooja/ram_sandbox/src/cross/data/captions.json",
                        help="captions.json produced by caption.py (optional)")
    parser.add_argument("--output",
                        default="results/test_results.json",
                        help="Output JSON path")
    parser.add_argument("--top-k",        type=int, default=10, dest="top_k",
                        help="Candidates passed from Stage 1 to Stage 3")
    parser.add_argument("--llm-batch",    type=int, default=8, dest="llm_batch",
                        help="Batch size for Stage 3 LLM forward passes")
    parser.add_argument("--stage1-only",  action="store_true", dest="stage1_only",
                        help="Stop after Stage 1 (fast sanity check)")
    parser.add_argument("--no-metrics",   action="store_true", dest="no_metrics",
                        help="Skip metrics (use when test JSON has no label_idx)")
    parser.add_argument(
        "--zero-shot-llm",
        action="store_true",
        dest="zero_shot_llm",
        help="Run Stage 3 with base Qwen model (no LoRA) for zero-shot eval"
    )
    parser.add_argument(
        "--entity-aug",
        default="",
        dest="entity_aug",
        help="Path to entity summaries JSON (optional)"
    )
    parser.add_argument(
        "--meta",
        type=str,
        required=True,
        help="Path to the structured output metadata JSON (kb_entity_desc.json)"
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out: Dict = {}

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    stage1_results = run_stage1(
        test_json=args.test,
        stage1_ckpt=args.stage1_ckpt,
        top_k=args.top_k,
    )
    out["stage1_topk"] = stage1_results

    if not args.no_metrics:
        s1_metrics = compute_metrics(stage1_results, use_stage1=True)
        out["metrics_stage1"] = s1_metrics
        logger.info("\n=== Stage 1 metrics ===")
        for k, v in s1_metrics.items():
            logger.info(
                f"  {k:12s}: {v:.4f}" if isinstance(v, float) else f"  {k:12s}: {v}"
            )

    if args.stage1_only or (args.lora_path is None and not args.zero_shot_llm):
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved → {args.output}")
        return

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    final_results = run_stage3(
        stage1_results=stage1_results,
        test_json=args.test,
        captions_json=args.captions,
        entity_aug_path=args.entity_aug if args.entity_aug else None,
        lora_path=args.lora_path,
        llm_model=args.llm_model,
        top_k=args.top_k,
        batch_size=args.llm_batch,
        meta_path=args.meta,
    )
    stage3_lookup = {r["mention_id"]: r for r in final_results}

    # Build FULL results (true pipeline)
    full_results = []
    for s1 in stage1_results:
        mid = s1["mention_id"]

        if mid in stage3_lookup:
            full_results.append(stage3_lookup[mid])
        else:
            full_results.append({
                "mention_id": mid,
                "label_entity_id": s1.get("label_entity_id"),
                "ranked_entity_ids": [],  # failure
            })

    out["final"] = full_results

    if not args.no_metrics:
    # ✅ Conditional (old behavior)
        conditional_metrics = compute_metrics(final_results, use_stage1=False)

    # ✅ True pipeline (new)
        combined_metrics = compute_metrics(full_results, use_stage1=False)

        out["metrics_stage1_3_conditional"] = conditional_metrics
        out["metrics_stage1_3_full"] = combined_metrics

        logger.info("\n=== Results ===")
        logger.info("\n--- Conditional (label in top-K) ---")
        print_comparison(s1_metrics, conditional_metrics)

        logger.info("\n--- Full pipeline (true) ---")
        print_comparison(s1_metrics, combined_metrics)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"\nFull results saved → {args.output}")


if __name__ == "__main__":
    main()
