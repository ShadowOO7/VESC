"""
train_pipeline.py — End-to-end training for the two-stage entity reranking system.

Stage 1: BAAI/bge-reranker-v2-m3 cross-encoder   (cross_encoder.py)
Stage 3: Qwen2.5-7B-Instruct + LoRA               (llm_reranker.py + llm_trainer.py)

Run order
─────────

STEP 0: Generate image captions (once, for Stage 3 LLM context)
  python caption.py \\
    --data      /home/kmpooja/ram_sandbox/src/prompt2/wikidiverse_top50_ready_train.json \\
                /home/kmpooja/ram_sandbox/src/prompt2/wikidiverse_top50_ready_valid.json \\
    --image-dir /home/kmpooja/UniMEL/wikidiverse/wiki/kb_image/kb_image \\
    --output    /home/kmpooja/ram_sandbox/src/cross/data/captions.json \\
    --model llava

STEP 1A: [Optional] Fine-tune Stage 1 cross-encoder (~20 min)
  python train_pipeline.py --stage 1

STEP 1B: Generate Stage 1 top-K files for Stage 3 training
  # Zero-shot (already ~88% R@1, skip fine-tuning):
  python train_pipeline.py --generate-topk

  # After fine-tuning:
  python train_pipeline.py --generate-topk \\
      --stage1-ckpt /home/kmpooja/ram_sandbox/results/cross_encoder/best/

STEP 2: Train Stage 3 LLM reranker
  python train_pipeline.py --stage 3 \\
      --llm-model /home/kmpooja/UniMEL/models/Qwen2-7B-Instruct

STEP 3: Evaluate
  python train_pipeline.py --eval \\
      --stage1-ckpt /home/kmpooja/ram_sandbox/results/cross_encoder/best/ \\
      --lora-path   /home/kmpooja/ram_sandbox/results/llm_stage/lora_best \\
      --llm-model   /home/kmpooja/UniMEL/models/Qwen2-7B-Instruct

Check retrieval ceiling at any time:
  python train_pipeline.py --check-ceiling

Expected recall progression
────────────────────────────
Retrieval baseline (no reranking):        R@1=72.8%  R@10=94.3%
Stage 1 zero-shot (BGE, no fine-tuning):  R@1=76-80% R@10=93-95%
Stage 1 fine-tuned (BGE, 3 epochs):       R@1=82-87% R@10=93-95%
Stage 1 + Stage 3 (Qwen-7B LLM):         R@1=90-94% R@10=93-95%
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cross_encoder import CrossEncoderReranker
from llm_reranker_byme import LLMRerankDataset, LLMReranker
from llm_trainer import LLMStageTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (all absolute)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_JSON    = "/home/kmpooja/ram_sandbox/src/prompt2/richpedia_top50_ready_train.json"
VAL_JSON      = "/home/kmpooja/ram_sandbox/src/prompt2/richpedia_top50_ready_valid.json"
CAPTIONS_JSON = "/home/kmpooja/ram_sandbox/src/cross/data_Richpedia/captions_richpedia.json"
STAGE1_TOPK   = "/home/kmpooja/ram_sandbox/src/cross/data_Richpedia/stage1_topk_train_richpedia.json"
STAGE1_TOPK_V   = "/home/kmpooja/ram_sandbox/src/cross/data_Richpedia/stage1_topk_val_richpedia.json"
ENTITY_AUG_JSON = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/richpedia_entity_summaries.json"
OUTPUT_DIR    = "/home/kmpooja/ram_sandbox/results2/richpedia/"

# Rich mention metadata from EntityExtractionPipeline
# Keys per item: id, generated_type, generated_description, generated_tags
#MENTION_META_TRAIN = "/home/kmpooja/ram_sandbox/wikidiverse/results/structured_output_meta_train.json"
#MENTION_META_VAL   = "/home/kmpooja/ram_sandbox/wikidiverse/results/structured_output_meta_valid.json"
#MENTION_META_TEST  = "/home/kmpooja/ram_sandbox/wikidiverse/results/structured_output_meta_test.json"

MENTION_META_TRAIN = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/structured_output_meta_train_Richpedia.json"
MENTION_META_VAL   = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/structured_output_meta_valid_Richpedia.json"
MENTION_META_TEST  = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/structured_output_meta_test_RichpediaMEL.json"
# Stage 1 cross-encoder checkpoint directory
CROSS_ENC_DIR = f"{OUTPUT_DIR}cross_encoder/"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class _LLMCfg:
    """Minimal config carrying the fields LLMStageTrainer reads."""
    output_dir     = OUTPUT_DIR
    fp16           = True
    llm_lr         = 1e-5          # conservative — loss still falling at epoch 3
    llm_batch_size = 1             # small — Qwen-7B fills ~15GB
    llm_epochs     = 4             # 3→5: loss was still decreasing
    llm_top_k      = 10
    llm_lora_r     = 32            # 16→32: more capacity for hard disambiguation


LLM_CFG = _LLMCfg()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cross_encoder(ckpt_path: str = None) -> CrossEncoderReranker:
    """
    Return a CrossEncoderReranker.
    ckpt_path = fine-tuned directory → load from there.
    ckpt_path = None / missing      → zero-shot base model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt_path and Path(ckpt_path).exists():
        logger.info(f"Loading fine-tuned cross-encoder from {ckpt_path}")
        reranker            = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.fp16       = True
        reranker.max_length = 128
        reranker.device     = torch.device(device)
        reranker.load(ckpt_path)
    else:
        if ckpt_path:
            logger.warning(
                f"Cross-encoder checkpoint not found at {ckpt_path!r} — "
                "using zero-shot base model."
            )
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            device=device, fp16=True, max_length=128,
        )
    return reranker


# ─────────────────────────────────────────────────────────────────────────────
# STEP: CHECK RETRIEVAL CEILING
# ─────────────────────────────────────────────────────────────────────────────

def check_ceiling():
    logger.info("Checking retrieval ceiling (Recall@50)…")
    with open(VAL_JSON) as fh:
        data = json.load(fh)

    total, found = 0, 0
    for rec in data:
        label = rec.get("label_idx")
        if label is None:
            for i, c in enumerate(rec["candidates"]):
                if c.get("is_correct"):
                    label = i
                    break
        if label is not None and label < len(rec["candidates"]):
            found += 1
        total += 1

    ceiling = found / max(total, 1)
    logger.info(
        f"Retrieval ceiling: {ceiling:.4f} ({found}/{total}) "
        f"→ max achievable R@1 = {ceiling*100:.1f}%"
    )
    if ceiling < 0.95:
        logger.warning(
            "Ceiling below 95%. Even a perfect reranker can't reach 95% R@1. "
            "Consider improving upstream retrieval."
        )
    return ceiling


# ─────────────────────────────────────────────────────────────────────────────
# STEP: TRAIN STAGE 1 — fine-tune cross-encoder
# ─────────────────────────────────────────────────────────────────────────────

def train_cross_encoder():
    """
    Fine-tune BAAI/bge-reranker-v2-m3 on WikiDiverse.

    Training: listwise CE — all 50 (mention, entity_text) pairs in one forward
    pass per mention, correct candidate's logit pushed above the other 49.

    Zero-shot already gives R@1 ~76-80%, fine-tuning adds ~5-7pp on top.
    Checkpoints saved to CROSS_ENC_DIR/best/.
    """
    logger.info("=== Stage 1: Fine-tuning cross-encoder ===")
    set_seed()

    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=True, max_length=128,
    )

    logger.info("Zero-shot performance before fine-tuning:")
    for k, v in reranker.evaluate(VAL_JSON).items():
        logger.info(f"  {k:12s}: {v:.4f}")

    reranker.fine_tune(
        train_json=TRAIN_JSON,
        val_json=VAL_JSON,
        output_dir=CROSS_ENC_DIR,
        num_epochs=5,
        lr=2e-6,
        weight_decay=1e-2,
        warmup_ratio=0.06,
        num_candidates=50,
        log_every=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP: GENERATE STAGE 1 TOP-K  → feed into Stage 3 training
# ─────────────────────────────────────────────────────────────────────────────

def generate_topk(stage1_ckpt: str = None):
    """
    Score all 50 candidates per mention with the cross-encoder, save top-K files.
    LLMRerankDataset reads these files to build Stage 3 training prompts.
    """
    logger.info("=== Generating Stage 1 top-K results ===")

    reranker = _load_cross_encoder(stage1_ckpt)

    logger.info("Stage 1 validation metrics:")
    for k, v in reranker.evaluate(VAL_JSON).items():
        logger.info(f"  {k:12s}: {v:.4f}")

    for in_json, out_json in [
        (TRAIN_JSON, STAGE1_TOPK),
        (VAL_JSON,   STAGE1_TOPK_V),
    ]:
        reranker.rerank_json(json_path=in_json, output_path=out_json,
                             top_k=LLM_CFG.llm_top_k)

    logger.info("Stage 1 top-K generation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP: TRAIN STAGE 3 — Qwen LLM reranker
# ─────────────────────────────────────────────────────────────────────────────

def train_stage3(
    llm_model: str = "/home/kmpooja/UniMEL/models/Qwen2-7B-Instruct",
    lora_r:    int = 32,
):
    """
    Train Qwen2.5-7B-Instruct + LoRA on Stage 1 top-K results.
    Requires STAGE1_TOPK and STAGE1_TOPK_V — run generate_topk first.

    Key training details:
    - shuffle_candidates=True: prevents "always output 1" collapse
    - mention_meta_path: injects rich type/description/tags from pipeline
    - Curriculum (in llm_trainer): epoch 1 hard-only, then full dataset
    """
    logger.info("=== Stage 3: LLM reranker training ===")
    set_seed()

    for path, label in [(STAGE1_TOPK, "train"), (STAGE1_TOPK_V, "val")]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Stage 1 top-K file missing: {path}\n"
                "Run first: python train_pipeline.py --generate-topk"
            )

    train_ds = LLMRerankDataset(
        stage1_results_path=STAGE1_TOPK,
        captions_path=CAPTIONS_JSON,
        candidate_json_path=TRAIN_JSON,
        top_k=LLM_CFG.llm_top_k,
        shuffle_candidates=True,        # prevents position-bias collapse
        mention_meta_path=MENTION_META_TRAIN,
        entity_aug_path=ENTITY_AUG_JSON,
    )
    val_ds = LLMRerankDataset(
        stage1_results_path=STAGE1_TOPK_V,
        captions_path=CAPTIONS_JSON,
        candidate_json_path=VAL_JSON,
        top_k=LLM_CFG.llm_top_k,
        shuffle_candidates=False,       # stable order for eval
        mention_meta_path=MENTION_META_VAL,
        entity_aug_path=ENTITY_AUG_JSON,
    )

    llm = LLMReranker(
        model_name_or_path=llm_model,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        top_k=LLM_CFG.llm_top_k,
    )

    LLMStageTrainer(llm, train_ds, val_ds, LLM_CFG).train()


# ─────────────────────────────────────────────────────────────────────────────
# STEP: EVALUATE — Stage 1 alone, then combined Stage 1+3
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_twostage(
    stage1_ckpt: str = None,
    lora_path:   str = None,
    llm_model:   str = "/home/kmpooja/UniMEL/models/Qwen2-7B-Instruct",
):
    """
    Evaluate both stages independently and combined.
    Stages run sequentially so they don't compete for GPU memory.
    """
    logger.info("=== Two-stage evaluation ===")
    results_out  = Path(OUTPUT_DIR) / "eval_twostage.json"
    all_metrics: dict = {}

    # ── Stage 1 ────────────────────────────────────────────────────────────
    logger.info("--- Stage 1 (BGE cross-encoder) ---")
    reranker = _load_cross_encoder(stage1_ckpt)
    s1_metrics = reranker.evaluate(VAL_JSON, top_k_list=(1, 5, 10))
    all_metrics["stage1"] = s1_metrics
    for k, v in s1_metrics.items():
        logger.info(f"  {k:12s}: {v:.4f}")

    if not lora_path or not Path(lora_path).exists():
        logger.info("No LoRA path provided — skipping Stage 3.")
        with open(results_out, "w") as fh:
            json.dump(all_metrics, fh, indent=2)
        return

    # Generate Stage 1 top-K on val for Stage 3 input
    val_topk_tmp = str(Path(OUTPUT_DIR) / "eval_val_stage1_topk.json")
    reranker.rerank_json(VAL_JSON, output_path=val_topk_tmp, top_k=LLM_CFG.llm_top_k)

    del reranker
    torch.cuda.empty_cache()

    # ── Stage 3 ────────────────────────────────────────────────────────────
    logger.info("--- Stage 3 (Qwen LLM reranker) ---")
    llm = LLMReranker.load_lora(
        base_model_name=llm_model, lora_path=lora_path, top_k=LLM_CFG.llm_top_k
    )

    val_ds = LLMRerankDataset(
        stage1_results_path=val_topk_tmp,
        captions_path=CAPTIONS_JSON,
        candidate_json_path=VAL_JSON,
        top_k=LLM_CFG.llm_top_k,
        shuffle_candidates=False,
        mention_meta_path=MENTION_META_VAL,
    )
    val_loader = DataLoader(
        val_ds, batch_size=LLM_CFG.llm_batch_size, shuffle=False,
        collate_fn=LLMRerankDataset.collate, num_workers=0,
    )

    recalls = {1: 0, 5: 0, 10: 0}
    rr_sum  = 0.0
    total   = 0

    llm.model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Stage 3 eval", dynamic_ncols=True):
            out    = llm(prompts=batch["prompts"], label_idx=batch["label_idx"])
            ranked = out["digit_logits"].argsort(dim=-1, descending=True)
            labels = batch["label_idx"].to(out["digit_logits"].device)

            for i in range(labels.shape[0]):
                match = (ranked[i] == labels[i]).nonzero(as_tuple=True)[0]
                rank  = match[0].item() + 1 if len(match) > 0 else LLM_CFG.llm_top_k + 1
                for k in recalls:
                    if rank <= k:
                        recalls[k] += 1
                rr_sum += 1.0 / rank
                total  += 1

    s3_metrics = {f"recall@{k}": recalls[k] / max(total, 1) for k in recalls}
    s3_metrics["mrr"] = rr_sum / max(total, 1)
    all_metrics["stage1+3"] = s3_metrics

    logger.info(f"\n{'Metric':15s}  {'Stage 1':>10}  {'Stage 1+3':>10}  {'Delta':>9}")
    logger.info("-" * 52)
    for k in ["recall@1", "recall@5", "recall@10", "mrr"]:
        v1  = s1_metrics.get(k, 0)
        v12 = s3_metrics.get(k, 0)
        logger.info(f"{k:15s}  {v1:>10.4f}  {v12:>10.4f}  {v12-v1:>+9.4f}")

    with open(results_out, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info(f"Results saved → {results_out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage entity reranking (BGE cross-encoder + Qwen LLM)"
    )

    parser.add_argument("--stage",         type=int, choices=[1, 3], default=1)
    parser.add_argument("--generate-topk", action="store_true", dest="generate_topk",
                        help="Generate Stage 1 top-K files for Stage 3 training")
    parser.add_argument("--eval",          action="store_true",
                        help="Evaluate Stage 1 + combined Stage 1+3 on val set")
    parser.add_argument("--check-ceiling", action="store_true", dest="check_ceiling")

    parser.add_argument("--stage1-ckpt",
                        default=f"{OUTPUT_DIR}cross_encoder/best/",
                        dest="stage1_ckpt",
                        help="Fine-tuned cross-encoder directory (omit → zero-shot)")
    parser.add_argument("--lora-path",
                        default=f"{OUTPUT_DIR}llm_stage/lora_best",
                        dest="lora_path",
                        help="Qwen LoRA checkpoint directory (for --eval)")
    parser.add_argument("--llm-model",
                        default="/home/kmpooja/UniMEL/models/Qwen2-7B-Instruct",
                        dest="llm_model")
    parser.add_argument("--lora-r",  type=int, default=32, dest="lora_r")

    args = parser.parse_args()

    if args.check_ceiling:
        check_ceiling()
    elif args.generate_topk:
        generate_topk(args.stage1_ckpt)
    elif args.eval:
        evaluate_twostage(args.stage1_ckpt, args.lora_path, args.llm_model)
    elif args.stage == 1:
        train_cross_encoder()
    elif args.stage == 3:
        train_stage3(args.llm_model, args.lora_r)
