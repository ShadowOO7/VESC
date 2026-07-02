import json, os, torch, transformers
from tqdm import tqdm

MODEL_DIR = "/home/kmpooja/UniMEL/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct"
DATA_FILE = "/home/kmpooja/UniMEL/RichpediaMEL/kb_entity.json"
OUT_FILE  = "/home/kmpooja/ram_sandbox/RichpediaMEL/results/richpedia_entity_summaries.json"

BATCH_SIZE = 8   # increase if GPU memory allows

with open(DATA_FILE) as f:
    entities = json.load(f)

# Resume logic
try:
    with open(OUT_FILE) as f:
        out_list = json.load(f)

    processed_ids = set(
        o["ids"] for o in out_list if o.get("sum", "")
    )

    print(f"Resuming: {len(processed_ids)} done")

except:
    out_list = []
    processed_ids = set()

pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_DIR,
    model_kwargs={"dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
pipeline.tokenizer.padding_side = "left"

PROMPT = (
    "Please generate a one-sentence summary for the given entity, "
    "including entity name and description.\n\n"
    "Entity name: {entity_name}\n"
    "Entity description: {entity_des}\n\n"
    "Generate a short summary in one concise factual sentence.\nSummary:"
)

# -------------------------------------------------------------------
# FILTER REMAINING ENTITIES
# -------------------------------------------------------------------

remaining = [
    e for e in entities
    if e.get("id") not in processed_ids
]

print(f"Remaining: {len(remaining)}")

# -------------------------------------------------------------------
# BATCH LOOP
# -------------------------------------------------------------------

for start in tqdm(range(0, len(remaining), BATCH_SIZE)):

    batch = remaining[start:start + BATCH_SIZE]

    prompts = []
    batch_records = []

    for e in batch:

        eid = e.get("id")
        name = e.get("entity_name", "")
        desc = e.get("desc", "")

        record = {
            "ids": eid,
            "qid": e.get("qid"),
            "name": name,
            "des": desc,
        }

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": PROMPT.format(
                    entity_name=name,
                    entity_des=desc
                )
            },
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append(prompt)
        batch_records.append(record)

    # ---------------------------------------------------------------
    # GENERATE WHOLE BATCH
    # ---------------------------------------------------------------

    try:

        outputs = pipeline(
            prompts,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            batch_size=BATCH_SIZE,
            pad_token_id=pipeline.model.config.eos_token_id,
            return_full_text=False,
        )

        for record, output in zip(batch_records, outputs):

            try:
                gen = output[0]["generated_text"].strip()

                if "Summary:" in gen:
                    gen = gen.split("Summary:")[-1].strip()

                gen = " ".join(gen.split())

                dot = gen.find(".")
                one_sent = (
                    gen[:dot + 1]
                    if 0 <= dot < 400
                    else gen[:300]
                )

                if record["name"] not in one_sent:
                    one_sent = f'{record["name"]} — {one_sent}'

                record["sum"] = one_sent

            except Exception as ex:
                print(f"Postprocess error: {ex}")
                record["sum"] = ""

            out_list.append(record)

    except Exception as ex:

        print(f"Batch error: {ex}")

        for record in batch_records:
            record["sum"] = ""
            out_list.append(record)

    # ---------------------------------------------------------------
    # SAVE PERIODICALLY
    # ---------------------------------------------------------------

    if start % (BATCH_SIZE * 20) == 0:
        with open(OUT_FILE, "w") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)

# FINAL SAVE
with open(OUT_FILE, "w") as f:
    json.dump(out_list, f, ensure_ascii=False, indent=2)

print(f"Done. {len(out_list)} entities saved.")
