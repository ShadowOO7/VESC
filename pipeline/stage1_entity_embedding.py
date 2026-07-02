import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

MODEL_DIR   = "/home/kmpooja/UniMEL/models/SFR-Embedding-Mistral/SFR-Embedding-Mistral"
DATA_FILE   = "/home/kmpooja/UniMEL/RichpediaMEL/kb_entity.json"
OUTPUT_FILE = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/entity_embedding.json"
MAX_LENGTH  = 4096
CHECKPOINT_EVERY = 500


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last non-padding token for each sequence."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def main():
    print(f"Loading entity data from: {DATA_FILE}")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total entities: {len(data)}")

    # Load Richpedia entity summaries and inject into data
    SUMMARY_FILE = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/richpedia_entity_summaries.json"
    import os as _os
    if _os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE) as sf:
            summ_list = json.load(sf)
        summ_map = {str(e["ids"]): e.get("sum", "") for e in summ_list}
        for e in data:
            eid = str(e.get("id", ""))
            if eid in summ_map and summ_map[eid]:
                e["sum"] = summ_map[eid]
            else:
                e["sum"] = e.get("desc", "")
        print(f"Injected summaries for entities")

    # ----------------------------------------------------------------
    # Resume: skip already embedded qids if output file exists
    # ----------------------------------------------------------------
    processed_qids = set()
    embeds = []

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            embeds = existing
            processed_qids = {e["qid"] for e in existing}
            print(f"Resuming: {len(processed_qids)} entities already embedded, skipping them.")
        except Exception as ex:
            print(f"Could not load existing output file ({ex}), starting from scratch.")
            embeds = []
    else:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"Loading embedding model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    model.to("cuda").eval()
    print("Model loaded.")

    # ----------------------------------------------------------------
    # Track only NEW embeddings added this run for correct checkpointing
    # ----------------------------------------------------------------
    new_count = 0

    for ent in tqdm(data, desc="Embedding entities"):
        qid = ent.get("qid")
        if qid is None:
            continue
        if qid in processed_qids:
            continue

        name = ent.get("name", "")
        summ = ent.get("sum", "")
        text = (name + ": " + summ).replace("\n", " ").strip()

        batch = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**batch)
            emb = last_token_pool(
                outputs.last_hidden_state,
                batch["attention_mask"]
            )[0]
            emb = F.normalize(emb, dim=0)

        embeds.append({
            "qid": qid,
            "emb": emb.tolist()
        })
        processed_qids.add(qid)
        new_count += 1

        # Checkpoint every N *new* embeddings, regardless of resume offset
        if new_count % CHECKPOINT_EVERY == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(embeds, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved — {new_count} new | {len(embeds)} total / {len(data)} entities.")

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embeds, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Saved {len(embeds)} entity embeddings to: {OUTPUT_FILE}")
    print(f"New embeddings added this run: {new_count}")


if __name__ == "__main__":
    main()
