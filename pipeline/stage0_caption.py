"""
caption.py — Visual captioning bridge using LLaVA v1.6.
Includes Auto-Resume Checkpointing and True GPU Batching.

Usage
─────
python caption.py \
    --data /home/kmpooja/ram_sandbox/src/prompt2/WikiMEL_top50_ready_train.json \
    --mention-image-dir /home/kmpooja/UniMEL/WikiMEL/mention_image \
    --entity-image-dir /home/kmpooja/UniMEL/WikiMEL/kb_image \
    --output /home/kmpooja/ram_sandbox/src/cross/data/captions_wikimel.json \
    --batch-size 16
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class LLaVACaptioner:
    """Wraps LLaVA-v1.6-Mistral-7B for richer captions."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        logger.info(f"Loading LLaVA-v1.6 from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            # attn_implementation="flash_attention_2" # Leave commented out unless installed
        )
        if device != "cuda":
            self.model = self.model.to(device)

        self.model.eval()
        self.device = device

    @torch.no_grad()
    def caption_batch(
        self,
        images: List[Image.Image],
        prompt: str = "[INST] <image>\nDescribe the person or main subject in detail. [/INST]",
    ) -> List[str]:
        # TRUE BATCHING: Duplicate prompt for every image
        prompts = [prompt] * len(images)

        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )

        decoded_texts = self.processor.batch_decode(out, skip_special_tokens=True)

        captions = []
        for text in decoded_texts:
            if "[/INST]" in text:
                text = text.split("[/INST]")[-1].strip()
            captions.append(text)

        return captions


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def collect_image_items(data_paths: List[str]) -> tuple:
    """Return dicts: mention_id → image_path, entity_id → image_path."""
    mention_images: Dict[str, str] = {}
    entity_images:  Dict[str, str] = {}

    for path in data_paths:
        with open(path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for rec in records:
            mid = rec["mention_id"]
            if mid not in mention_images:
                mention_images[mid] = rec.get("mention_image", "")

            for c in rec.get("candidates", []):
                eid = c["entity_id"]
                if eid not in entity_images:
                    # Force the Q-format for entities
                    entity_images[eid] = f"{eid}_0.jpg"

    return mention_images, entity_images


def _load_image(rel: str, image_dir: Path) -> Optional[Image.Image]:
    if not rel:
        return None

    p = Path(rel)
    if p.is_absolute():
        if p.exists():
            try: return Image.open(p).convert("RGB")
            except Exception as e:
                logger.warning(f"⚠️ IMAGE ERROR: Could not read {p} | Error: {e}")
                return None
        return None

    filename = p.name

    # 1. Generate filename variations
    name_variations = [filename]

    if filename.startswith("mention_"):
        num_part = filename.replace("mention_", "")
        name_variations.extend([
            f"train_{num_part}",
            f"test_{num_part}",
            f"valid_{num_part}",
            f"val_{num_part}"
        ])
    elif filename.endswith("_0.jpg"):
        base_qid = filename.replace("_0.jpg", "")
        name_variations.extend([
            f"{base_qid}_1.jpg",
            f"{base_qid}_2.jpg",
            f"{base_qid}_3.jpg",
            f"{base_qid}_4.jpg",
            f"{base_qid}_5.jpg",
            f"{base_qid}_6.jpg",
            f"{base_qid}_7.jpg",
            f"{base_qid}_8.jpg",
            f"{base_qid}_9.jpg"
        ])
    # ADD HERE ↓ (after full if/elif block)
    name_variations += [n.lower() for n in name_variations]
    name_variations = list(dict.fromkeys(name_variations))  # deduplicate

    # 2. Subdirectories to check (handles wikidiverse quirks)
    dirs_to_check = [
        image_dir,
        image_dir / "train_image",
        image_dir / "test_image",
        image_dir / "valid_image",
    ]

    # 3. Hunt for the image
    for d in dirs_to_check:
        for name_var in name_variations:
            full_path = d / name_var
            if full_path.exists():
                try:
                    return Image.open(full_path).convert("RGB")
                except Exception as e:
                    logger.warning(f"⚠️ IMAGE ERROR: Could not read {full_path} | Error: {e}")
                    return None

    logger.warning(f"❌ IMAGE NOT FOUND: Looked for {filename} in {image_dir}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING & PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(output_path: Path) -> dict:
    """Load existing captions to resume training safely."""
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.info(f"✅ Found Checkpoint! Resuming from {len(data.get('mentions', {}))} mentions and {len(data.get('entities', {}))} entities.")
            return data
        except Exception as e:
            logger.warning(f"⚠️ Could not read checkpoint {output_path}: {e}. Starting fresh.")
    return {"mentions": {}, "entities": {}}


def save_checkpoint(output_path: Path, data: dict):
    """Safely save checkpoint using a temporary file to prevent corruption."""
    temp_path = output_path.with_suffix(".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        
    temp_path.replace(output_path)  # Atomic replace


def caption_all(
    id_to_path: Dict[str, str],
    image_dir:  Path,
    captioner:  LLaVACaptioner,
    existing_captions: Dict[str, str],
    save_fn,
    batch_size: int = 16,
    prefix:     str = "",
) -> Dict[str, str]:
    """Run captioning, skipping items that are already in existing_captions."""
    
    # 1. Filter out items we've already processed
    ids_to_process = [iid for iid in id_to_path.keys() if iid not in existing_captions]
    total = len(ids_to_process)
    
    if total == 0:
        logger.info(f"  {prefix}: All items already captioned! Skipping.")
        return existing_captions
        
    logger.info(f"  {prefix}: {total} remaining items to process...")

    # 2. Process remaining items
    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        chunk = ids_to_process[start:end]

        images, valid_ids = [], []
        for iid in chunk:
            img = _load_image(id_to_path[iid], image_dir)
            if img is not None:
                images.append(img)
                valid_ids.append(iid)
            else:
                existing_captions[iid] = ""   # Save empty string for missing images so we don't retry them

        if images:
            try:
                texts = captioner.caption_batch(images)
                for iid, text in zip(valid_ids, texts):
                    existing_captions[iid] = text.strip()
            except Exception as e:
                logger.warning(f"Caption batch failed: {e}")
                for iid in valid_ids:
                    existing_captions[iid] = ""

        # 3. Print log and save checkpoint every 50 batches
        batches_done = (start // batch_size) + 1
        if batches_done % 20 == 0:
            logger.info(f"  {prefix}: Processed {end}/{total} remaining items")
            
        if batches_done % 50 == 0:
            logger.info(f"💾 Saving checkpoint at {end} items...")
            save_fn()

    return existing_captions


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate image captions for entity linking")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--mention-image-dir", required=True, help="Path to mention images")
    parser.add_argument("--entity-image-dir", required=True, help="Path to KB entity images")
    parser.add_argument("--output", default="data/captions.json")
    parser.add_argument("--model-path", default="/home/kmpooja/UniMEL/models/llava-v1.6-mistral-7b-hf/llava-v1.6-mistral-7b-hf", dest="model_path")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    mention_dir = Path(args.mention_image_dir)
    entity_dir = Path(args.entity_image_dir)
    output_file = Path(args.output)

    # Load existing checkpoint data
    master_data = load_checkpoint(output_file)
    
    # Helper function to save current state
    def save_current_state():
        save_checkpoint(output_file, master_data)

    captioner = LLaVACaptioner(args.model_path, args.device)

    logger.info("Collecting image paths...")
    mention_images, entity_images = collect_image_items(args.data)
    logger.info(f"  Dataset contains {len(mention_images)} mentions, {len(entity_images)} entities total.")

    logger.info("--- Captioning Mention Images ---")
    master_data["mentions"] = caption_all(
        mention_images, mention_dir, captioner, master_data["mentions"], save_current_state, args.batch_size, prefix="mentions"
    )
    save_current_state() # Save after mentions are fully done

    logger.info("--- Captioning Entity Images ---")
    master_data["entities"] = caption_all(
        entity_images, entity_dir, captioner, master_data["entities"], save_current_state, args.batch_size, prefix="entities"
    )
    save_current_state() # Final save

    logger.info(f"🎉 Pipeline Complete! Captions fully saved to → {args.output}")


if __name__ == "__main__":
    main()
