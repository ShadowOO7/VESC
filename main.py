"""
VSCE-EL: Visual-Semantic Cross-Encoder Entity Linking
Full pipeline runner.

Usage:
    bash run.sh richpedia          # full pipeline
    bash run.sh wikimel eval       # evaluation only
    bash run.sh wikidiverse 3      # cross-encoder stage only

    python main.py --config configs/richpedia.yaml --stage all
    python main.py --config configs/richpedia.yaml --stage 0   # mention enhancement
    python main.py --config configs/richpedia.yaml --stage 1   # entity summary+embedding
    python main.py --config configs/richpedia.yaml --stage 2   # hybrid retrieval
    python main.py --config configs/richpedia.yaml --stage 3   # cross-encoder train+topk
    python main.py --config configs/richpedia.yaml --stage 4   # LLM reranker train
    python main.py --config configs/richpedia.yaml --stage eval # evaluate on test set
"""

import argparse, yaml, os, sys, subprocess
from pathlib import Path
from types import SimpleNamespace


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    def to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
        return obj
    return to_ns(raw)


def ensure_dirs(cfg):
    for key, val in vars(cfg.paths).items():
        if isinstance(val, str) and (val.endswith('/') or 'ckpt' in key):
            Path(val).mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)


def run_stage(script, args_list):
    """Run a pipeline stage as subprocess using current python."""
    python = sys.executable
    cmd = [python, script] + args_list
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    return result.returncode


# ── Individual stage functions ────────────────────────────────────────────────

def stage0(cfg):
    print("\n" + "="*60)
    print("STAGE 0: Mention Enhancement + LLaVA Captions")
    print("="*60)
    for split in ["train", "valid", "test"]:
        meta_out = getattr(cfg.paths, f"meta_{split}")
        run_stage("pipeline/stage0_mention_enhancement.py", [
            "--split", split,
            "--kb",    cfg.paths.kb_file,
            "--images", cfg.paths.mention_images,
            "--qwen",  cfg.paths.qwen_model,
            "--ram",   cfg.paths.ram_model,
            "--out",   meta_out,
        ])
    run_stage("pipeline/stage0_caption.py", [
        "--images",  cfg.paths.mention_images,
        "--llava",   cfg.paths.llava_model,
        "--out",     cfg.paths.captions,
    ])
    print("Stage 0 done ✅")


def stage1(cfg):
    print("\n" + "="*60)
    print("STAGE 1: Entity Summary + SFR Embedding")
    print("="*60)
    run_stage("pipeline/stage1_entity_summary.py", [
        "--kb",    cfg.paths.kb_file,
        "--llama", cfg.paths.llama_model,
        "--out",   cfg.paths.entity_summaries,
    ])
    run_stage("pipeline/stage1_entity_embedding.py", [
        "--summaries", cfg.paths.entity_summaries,
        "--sfr",       cfg.paths.sfr_model,
        "--out",       cfg.paths.entity_embeddings,
    ])
    print("Stage 1 done ✅")


def stage2(cfg):
    print("\n" + "="*60)
    print("STAGE 2: Hybrid Retrieval (Top-50)")
    print("="*60)
    for split in ["train", "valid", "test"]:
        meta_path   = getattr(cfg.paths, f"meta_{split}")
        hybrid_out  = getattr(cfg.paths, f"hybrid_{split}")
        ready_out   = getattr(cfg.paths, f"ready_{split}")
        run_stage("pipeline/stage2_hybrid_retrieval.py", [
            "--meta",     meta_path,
            "--ent-emb",  cfg.paths.entity_embeddings,
            "--ent-sum",  cfg.paths.entity_summaries,
            "--alpha",    str(cfg.hybrid.alpha),
            "--top-k",    str(cfg.hybrid.top_k),
            "--out",      hybrid_out,
        ])
        run_stage("pipeline/stage2_join_mention.py", [
            "--hybrid",  hybrid_out,
            "--meta",    meta_path,
            "--out",     ready_out,
        ])
    print("Stage 2 done ✅")


def stage3(cfg):
    print("\n" + "="*60)
    print("STAGE 3: Cross-Encoder Fine-tuning + Top-10 Generation")
    print("="*60)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_stage("train.py", [
        "--stage", "1",
        "--train", cfg.paths.ready_train,
        "--val",   cfg.paths.ready_valid,
        "--out",   cfg.paths.cross_encoder_ckpt,
    ])
    run_stage("train.py", [
        "--generate-topk",
        "--stage1-ckpt",    cfg.paths.cross_encoder_ckpt,
        "--topk-train-out", cfg.paths.stage1_topk_train,
        "--topk-val-out",   cfg.paths.stage1_topk_val,
    ])
    print("Stage 3 done ✅")


def stage4(cfg):
    print("\n" + "="*60)
    print("STAGE 4: LLM Reranker Fine-tuning (Qwen2 + LoRA)")
    print("="*60)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_stage("train.py", [
        "--stage", "3",
        "--stage1-ckpt", cfg.paths.cross_encoder_ckpt,
        "--llm-model",   cfg.paths.qwen_model,
        "--out",         cfg.paths.lora_ckpt,
    ])
    print("Stage 4 done ✅")


def stage_eval(cfg):
    print("\n" + "="*60)
    print("EVALUATION: Full Pipeline on Test Set")
    print("="*60)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_stage("evaluate.py", [
        "--test",        cfg.paths.ready_test,
        "--meta",        cfg.paths.meta_test,
        "--stage1-ckpt", cfg.paths.cross_encoder_ckpt,
        "--lora-path",   cfg.paths.lora_ckpt,
        "--llm-model",   cfg.paths.qwen_model,
        "--captions",    cfg.paths.captions,
        "--entity-aug",  cfg.paths.entity_summaries,
        "--output",      cfg.paths.results,
    ])
    print(f"\nResults → {cfg.paths.results} ✅")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VSCE-EL Pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="all",
                        choices=["all","0","1","2","3","4","eval"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    print(f"\n{'='*60}")
    print(f"VSCE-EL  |  Dataset: {cfg.dataset.upper()}  |  Stage: {args.stage}")
    print(f"{'='*60}")

    stages = {"0":stage0,"1":stage1,"2":stage2,"3":stage3,"4":stage4,"eval":stage_eval}

    if args.stage == "all":
        for fn in stages.values():
            fn(cfg)
    else:
        stages[args.stage](cfg)


if __name__ == "__main__":
    main()
