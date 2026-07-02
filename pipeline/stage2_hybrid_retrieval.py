import json
import numpy as np
from tqdm import tqdm
from rapidfuzz import fuzz

# =========================
# PATHS
# =========================

QUERY_FILE = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/structured_output_embeddings_test_RichpediaMEL.json"
ENTITY_EMB_FILE = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/entity_embedding.json"
KB_FILE = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/richpedia_entity_summaries.json"

ALPHA = 0.2
TOP_KS = [10, 50, 100]

# =========================
# LOAD KB
# =========================

print("Loading KB...")
with open(KB_FILE, "r", encoding="utf-8") as f:
    kb = json.load(f)

# Richpedia/WikiMEL often uses "entity_name" or "name". We check both.
entity_name_map = {}
for e in kb:
    qid = e.get("qid")
    name = e.get("entity_name", e.get("name", ""))
    if qid:
        entity_name_map[qid] = str(name).lower()

# =========================
# LOAD ENTITY EMBEDDINGS
# =========================

print("Loading entity embeddings...")
with open(ENTITY_EMB_FILE, "r", encoding="utf-8") as f:
    entity_data = json.load(f)

entity_qids = []
entity_names = []
entity_embs = []

for e in entity_data:
    qid = e.get("qid")
    if qid in entity_name_map:
        entity_qids.append(qid)
        entity_names.append(entity_name_map[qid])
        entity_embs.append(e["emb"])

entity_embs = np.array(entity_embs)
print(f"Entities loaded: {len(entity_qids)}")

# =========================
# LOAD QUERY DATA
# =========================

print("Loading queries...")
with open(QUERY_FILE, "r", encoding="utf-8") as f:
    query_data = json.load(f)

query_ids = []
query_embs = []
mentions = []
answers = []

for q in query_data:
    # Richpedia query format uses "id", "mentions", and "answer"
    q_id = q.get("id", q.get("mention_id"))
    query_ids.append(q_id)

    query_embs.append(q.get("emb", q.get("embedding")))

    # Extract the mention string
    m_text = q.get("mentions", q.get("mention", ""))
    mentions.append(str(m_text).lower())

    # Extract the ground truth QID
    ans = q.get("answer", q.get("qid"))
    answers.append(ans)

query_embs = np.array(query_embs)
print(f"Queries loaded: {len(query_ids)}")

# =========================
# NORMALIZE EMBEDDINGS
# =========================

entity_embs = entity_embs / np.linalg.norm(entity_embs, axis=1, keepdims=True)
query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)

# =========================
# HYBRID RETRIEVAL
# =========================

results = {k: [] for k in TOP_KS}
recall_hits = {k: 0 for k in TOP_KS}

print("Running hybrid retrieval...")

for i in tqdm(range(len(query_ids))):

    mention_vec = query_embs[i]
    mention_text = mentions[i]

    # -------------------
    # COSINE SIMILARITY (vectorized)
    # -------------------
    cosine_scores = np.dot(entity_embs, mention_vec)

    # -------------------
    # FUZZY SCORES
    # -------------------
    # Rapidfuzz is fast, but doing this N times inside a loop can be heavy.
    # It will take a minute or two to complete depending on KB size!
    fuzzy_scores = np.array([
        fuzz.ratio(mention_text, name) / 100.0
        for name in entity_names
    ])

    # -------------------
    # HYBRID SCORE
    # -------------------
    hybrid_scores = ALPHA * fuzzy_scores + (1 - ALPHA) * cosine_scores

    sorted_idx = np.argsort(hybrid_scores)[::-1]

    for k in TOP_KS:
        top_idx = sorted_idx[:k]
        top_qids = [entity_qids[j] for j in top_idx]

        # Check for Recall
        if answers[i] in top_qids:
            recall_hits[k] += 1

        # Format exactly how the downstream LLM expects it
        results[k].append({
            "id": query_ids[i],
            "answer": answers[i],
            "cands": top_qids
        })

# =========================
# PRINT RECALL
# =========================

print("\nRecall Results")
for k in TOP_KS:
    recall = recall_hits[k] / len(query_ids)
    print(f"Recall@{k}: {recall:.4f}")

# =========================
# SAVE OUTPUT
# =========================

for k in TOP_KS:
    out_file = f"/home/kmpooja/ram_sandbox/RichpediaMEL/results/hybrid_top{k}_alpha02_test.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results[k], f, indent=2)
    print("Saved:", out_file)

print("\nDone.")
