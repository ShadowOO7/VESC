import json
from tqdm import tqdm

# =========================
# PATHS
# =========================
KB_FILE   = "/home/kmpooja/UniMEL/RichpediaMEL/kb_entity.json"
SPLITS = [
    (
        "/home/kmpooja/ram_sandbox/RichpediaMEL/results/structured_output_meta_train_Richpedia.json",
        "/home/kmpooja/ram_sandbox/RichpediaMEL/results/hybrid_top50_alpha02_train.json",
        "/home/kmpooja/ram_sandbox/src/prompt2/richpedia_top50_ready_train.json",
    ),
    (
        "/home/kmpooja/ram_sandbox/RichpediaMEL/results/structured_output_meta_valid_Richpedia.json",
        "/home/kmpooja/ram_sandbox/RichpediaMEL/results/hybrid_top50_alpha02_valid.json",
        "/home/kmpooja/ram_sandbox/src/prompt2/richpedia_top50_ready_valid.json",
    ),
    (
        "/home/kmpooja/ram_sandbox/RichpediaMEL/results/structured_output_meta_test_RichpediaMEL.json",
        "/home/kmpooja/ram_sandbox/RichpediaMEL/results/hybrid_top50_alpha02_test.json",
        "/home/kmpooja/ram_sandbox/src/prompt2/richpedia_top50_ready_test.json",
    ),
]

# =========================
# LOAD KB
# =========================
print("Loading knowledge base...")
with open(KB_FILE) as f:
    kb = json.load(f)

entity_map = {}
for e in kb:
    qid = e["qid"]
    name = e.get("entity_name", "")
    desc = e.get("desc", "")
    images = e.get("image_list", [])
    img = images[0] if images else ""
    entity_map[qid] = {
        "entity_text": f"{name}. {desc}",
        "entity_image": img
    }
print(f"Entity map built: {len(entity_map)} entities")

# =========================
# PROCESS EACH SPLIT
# =========================
for meta_path, cand_path, out_path in SPLITS:
    split_name = out_path.split("/")[-1]
    print(f"\nProcessing {split_name}...")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(cand_path) as f:
        cands = json.load(f)

    # Build mention map — id → meta
    mention_map = {}
    for m in meta:
        m_id = str(m.get("id"))
        mention_map[m_id] = {
            "mention_text": f"{m.get('mentions','')} {m.get('sentence','')}".strip(),
            "mention_image": m.get("imgPath", ""),
            "answer": m.get("answer")
        }

    dataset = []
    skipped_no_meta = 0
    skipped_no_recall = 0

    for sample in tqdm(cands):
        m_id = str(sample.get("id"))

        if m_id not in mention_map:
            skipped_no_meta += 1
            continue

        ans_qid = mention_map[m_id]["answer"]
        candidates = sample.get("cands", [])

        cand_objs = []
        label_idx = None

        for cqid in candidates:
            ent = entity_map.get(cqid, {"entity_text": "", "entity_image": ""})
            cand_objs.append({
                "entity_id": cqid,
                "entity_text": ent["entity_text"],
                "entity_image": ent["entity_image"]
            })
            if cqid == ans_qid:
                label_idx = len(cand_objs) - 1

        if label_idx is None:
            skipped_no_recall += 1
            continue

        dataset.append({
            "mention_id": m_id,
            "mention_text": mention_map[m_id]["mention_text"],
            "mention_image": mention_map[m_id]["mention_image"],
            "label_idx": label_idx,
            "candidates": cand_objs
        })

    print(f"  Skipped (no meta): {skipped_no_meta}")
    print(f"  Skipped (not in top-50): {skipped_no_recall}")
    print(f"  Saved: {len(dataset)} samples")

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  → {out_path}")

print("\nDone.")
