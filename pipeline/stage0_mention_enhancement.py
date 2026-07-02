import sys
import os
import json
import torch
import numpy as np
import logging
import gc
import re
from difflib import SequenceMatcher
from PIL import Image
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    BitsAndBytesConfig
)

# ==========================================
# 1. LOCAL PATH CONFIGURATION
# ==========================================
BATCH_SIZE = 8  # Processing 12 items at a time

BASE_CONFIG = {
    "RECOGNIZE_ANYTHING_REPO": "/home/kmpooja/ram_sandbox/recognize-anything",
    "RAM_CKPT": "/home/kmpooja/ram_sandbox/models/ram_plus_swin_large_14m.pth",
    "QWEN_PATH": "/home/kmpooja/UniMEL/models/Qwen2-7B-Instruct",
    "SFR_PATH": "/home/kmpooja/UniMEL/models/SFR-Embedding-Mistral/SFR-Embedding-Mistral",
    "IMAGE_DIR": "/home/kmpooja/UniMEL/RichpediaMEL/mention_images"
}

SPLITS = [
    {
        "DATASET_FILE":    "/tmp/richpedia_missing_train.json",
        "OUTPUT_FILE_META": "/tmp/structured_output_meta_train_missing.json",
        "OUTPUT_FILE_EMB":  "/tmp/structured_output_embeddings_train_missing.json",
    },
    # valid and test already complete — skipped
    # {
    #     "DATASET_FILE":    "/home/kmpooja/UniMEL/RichpediaMEL/RichpediaMEL_dev.json",
    # },
    # {
    #     "DATASET_FILE":    "/home/kmpooja/UniMEL/RichpediaMEL/RichpediaMEL_test.json",
    # },
]

sys.path.append("/home/kmpooja/ram_sandbox/recognize-anything")
from ram.models.ram_plus import ram_plus
from ram.inference import inference_ram

# Prevent PyTorch from fragmenting memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# ==========================================
# 2. PROMPTS & CONSTANTS
# ==========================================

VALID_TYPES = {
    "person", "nationality", "religious group", "political", "weather",
    "health", "technology", "group", "organization", "country", "city",
    "state", "building", "airport", "highway", "education", "economy",
    "bridge", "company", "agency", "institution", "product", "event",
    "work of art", "crime", "sports", "entertain", "disaster", "law",
    "language", "location", "other"
}

TYPE_PROMPT = """
Entity mention: "{mention_name}"
Context sentence: "{mention_context}"
Image Object Tags: {image_tags}

Use the context sentence and provided image object tags together to disambiguate the mention correctly before identifying the type.

Important disambiguation examples:
- "Java" in "Jakarta is on the island of Java" → the island, NOT the programming language → type: location
- "Maffra (white shorts)" in a sports standings sentence → a sports club, NOT a city → type: sports
- "fruit" in a sentence about Durian → the specific fruit Durian → type: other
- "GL" in a football ladder sentence → the league (Gippsland Football League) → type: organization
- "Khan" in a sentence about nuclear technology and North Korea → a specific person → type: person
- "Polish" referring to nationality → the country Poland → type: country
- "Chinese" referring to nationality → the country China → type: country
- "area" used to refer to a specific geographic region → type: location
- If the type does not clearly fit any category → type: other

Task: Identify the entity type of "{mention_name}" ONLY — not of any other entity in the sentence.

Choose ONLY one from:
person, nationality, religious group, political, weather, health, technology, group, organization, country, city, state, building, airport, highway, education, economy, bridge, company, agency, institution, product, event, work of art, crime, sports, entertain, disaster, law, language, location, other

Output format STRICTLY (one line only, no hashtags, no extra text):
Type: [write the exact entity type here]

Answer:
"""

DESC_PROMPT = """
Entity mention: "{mention_name}"
Entity type: "{mention_type}"
Context sentence: "{mention_context}"
Image Object Tags: {image_tags}

Use the context sentence and provided image object tags together to disambiguate the mention correctly before generating the description.

Identify the correct real-world entity referred to, using the context sentence as the primary guide.

IMPORTANT: You must generate a description for the entity "{mention_name}" ITSELF, not for any other entity mentioned in the context sentence.
- Use the context sentence ONLY to disambiguate what real-world entity "{mention_name}" refers to.
- Then describe THAT entity using general world knowledge.
- Use the context sentence and tags as the PRIMARY source to identify the specific real-world entity.

Disambiguation rules:
1. If the mention is a nationality adjective or demonym:
   - Do NOT say "{mention_name} is a country" — the adjective itself is not the country.
   - Instead, describe what it refers to. Example: "Chinese refers to the people of China..."
2. If the mention is a generic word (e.g. "fruit", "area"):
   - Identify THAT specific entity first, then describe it.
3. For all other mentions:
   - Use context and tags to disambiguate and identify the real-world entity, then describe it from world knowledge.

Task: Write ONE clear factual sentence introducing the entity "{mention_name}".

Rules:
- The sentence MUST start with "{mention_name}".
- NEVER use phrases like "indicated by the tags", "the image shows", or "in the context". Use the tags silently in your head.
- Focus on occupation, nationality, role, or significance.
- Do NOT describe other entities present in the context sentence.
- Do NOT copy or rephrase the context sentence.
- Keep it between 10–25 words.
- Output ONLY the description sentence — no hashtags, no labels, no extra lines.
- The sentence MUST end with a period (.)

Output format STRICTLY (one line only):
Description: [write your 10 to 25 word sentence here]

Answer:
"""

# ==========================================
# 3. FILTERING & PARSING LOGIC
# ==========================================

def clean_type(ent_type):
    ent_type = ent_type.lower().strip()
    ent_type = ent_type.split("\n")[0].split(".")[0].strip()
    if "#" in ent_type:
        ent_type = ent_type[:ent_type.index("#")].strip()
    
    # Remove literal brackets if Qwen outputs them
    ent_type = ent_type.replace("[", "").replace("]", "").strip()
    
    if ent_type not in VALID_TYPES:
        return "other"
    return ent_type

def parse_type_output(text):
    type_match = re.search(r"Type:\s*(.*)", text, re.I)
    ent_type = type_match.group(1).strip() if type_match else "other"
    return clean_type(ent_type)

def parse_desc_output(text):
    desc_match = re.search(r"Description:\s*(.*)", text, re.I)
    desc = desc_match.group(1).strip() if desc_match else text
    return desc

def clean_description(desc, name):
    desc = desc.strip()
    
    # Remove literal brackets if Qwen outputs them
    desc = desc.replace("[", "").replace("]", "").strip()

    if "#" in desc:
        desc = desc[:desc.index("#")].strip()
    desc = re.sub(r'\s*#\s*\w.*$', '', desc, flags=re.S).strip()

    desc = re.sub(r"\b(sample output.*)", "", desc, flags=re.I)
    desc = re.sub(r"\b(my answer.*)", "", desc, flags=re.I)
    desc = re.sub(r"\b(answer\s*:)", "", desc, flags=re.I)
    desc = re.sub(r"\b(provide your.*)", "", desc, flags=re.I)
    desc = re.sub(r"\b(output format.*)", "", desc, flags=re.I)
    desc = re.sub(r"\b(note:.*)", "", desc, flags=re.I)
    desc = re.sub(r"\s+period\.?$", "", desc, flags=re.I)

    desc = re.sub(r"\s+", " ", desc)
    desc = re.sub(r"^(answer|response)\s*:\s*", "", desc, flags=re.I)
    desc = re.sub(r"\bC:\s*", "", desc)

    match = re.search(r'(?<![A-Z])\.(?:\s+[A-Z]|$)', desc)
    if match:
        desc = desc[:match.start() + 1]

    words = desc.split()
    if words:
        first_word = words[0].lower()
        name_lower = name.lower()
        name_no_poss = re.sub(r"'s?$", "", name_lower).strip()

        sim_full = SequenceMatcher(None, first_word, name_lower).ratio()
        sim_no_poss = SequenceMatcher(None, first_word, name_no_poss).ratio()

        if (name_lower != first_word
                and name_no_poss != first_word
                and sim_full < 0.8
                and sim_no_poss < 0.8):
            desc = name + " " + desc

    pattern = rf"^({re.escape(name)})\s+(?:the\s+)?{re.escape(name)}"
    desc = re.sub(pattern, r"\1", desc, flags=re.I)
    desc = re.sub(rf"({re.escape(name)} is a)\s+\1", r"\1", desc, flags=re.I)
    desc = re.sub(r"\ba ([aeiouAEIOU])", r"an \1", desc)
    desc = re.sub(r"\ban ([^aeiouAEIOU])", r"a \1", desc)

    desc = desc.strip()
    if desc and not desc.endswith(('.', '!', '?')):
        desc += "."

    return desc.strip()

# ==========================================
# 4. PIPELINE CLASS
# ==========================================

class EntityExtractionPipeline:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self._load_ram()
        self._load_dino()
        self._load_qwen()
        self._load_embedder()

    def _load_ram(self):
        logger.info("Loading RAM++")
        self.ram = ram_plus(pretrained=CONFIG["RAM_CKPT"], vit="swin_l", image_size=384)
        self.ram = self.ram.to(self.device)
        self.ram.eval()

    def _load_dino(self):
        logger.info("Loading DINO (Downloading from Hugging Face)")
        dino_repo_id = "IDEA-Research/grounding-dino-tiny"
        self.dino_processor = AutoProcessor.from_pretrained(dino_repo_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_repo_id)
        self.dino_model = self.dino_model.to(self.device)
        self.dino_model.eval()

    def _load_qwen(self):
        logger.info("Loading Qwen (4-bit)")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["QWEN_PATH"], local_files_only=True, trust_remote_code=True)
        self.tokenizer.padding_side="left"
        
        # Ensure tokenizer has a pad token for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.qwen = AutoModelForCausalLM.from_pretrained(
            CONFIG["QWEN_PATH"], 
            device_map="auto", 
            quantization_config=bnb_config, 
            local_files_only=True,
            trust_remote_code=True
        )
        self.qwen.eval()

    def _load_embedder(self):
        logger.info("Loading SFR-Mistral Embedding Model (in FP16)")
        self.embed_model = SentenceTransformer(
            CONFIG["SFR_PATH"], 
            trust_remote_code=True, 
            local_files_only=True,
            model_kwargs={"torch_dtype": torch.float16} # Crucial for 32GB GPU
        )
        self.embed_model = self.embed_model.to(self.device)
        self.embed_model.eval()

    # ---------------- Execution Methods ----------------
    def _load_local_image(self, image_path):
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Image load failed for {image_path}: {e}")
            return None

    def _get_tags(self, img):
        if img is None:
            return []
        
        arr = np.array(img)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(self.device)
        
        with torch.no_grad():
            ram_tags, _ = inference_ram(x, self.ram)
            
        tags = [t.strip() for t in (ram_tags.split(",") if isinstance(ram_tags, str) else ram_tags) if t.strip()]

        if not tags: 
            return tags
            
        text = ". ".join(tags) + "."
        inputs = self.dino_processor(images=img, text=text, return_tensors="pt").to(self.device)
        
        try:
            with torch.no_grad():
                out = self.dino_model(**inputs)
            res = self.dino_processor.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=0.3, text_threshold=0.25, target_sizes=[img.size[::-1]]
            )
            if res and res[0]["labels"]:
                tags = res[0]["labels"]
        except Exception as e:
            pass # DINO failed silently, fallback to RAM tags
            
        return tags

    def _generate_text_batched(self, prompts, max_tokens=80):
        if not prompts: return []
        
        formatted_prompts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": "You are a precise, academic vision-language analyst. Follow output formats exactly without repeating instructions."},
                {"role": "user", "content": p}
            ]
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.qwen.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                do_sample=True, 
                temperature=0.1, 
                top_p=0.9
            )
            
        input_length = inputs.input_ids.shape[1]
        results = []
        for out in outputs:
            results.append(self.tokenizer.decode(out[input_length:], skip_special_tokens=True).strip())
            
        del inputs
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return results

    def process_batch(self, batch_data):
        tags_list = []
        
        # 1. Sequential Tags
        for item in batch_data:
            img = self._load_local_image(item['img_path'])
            tags = self._get_tags(img)
            tags_list.append(tags)
            if img: del img
        gc.collect()

        # 2. Batch Types
        type_prompts = []
        for i, item in enumerate(batch_data):
            tag_text = ", ".join(tags_list[i]) if tags_list[i] else "None"
            type_prompts.append(TYPE_PROMPT.format(mention_name=item['mention_name'], mention_context=item['original_row'].get('sentence',''), image_tags=tag_text))
            
        raw_types = self._generate_text_batched(type_prompts, max_tokens=20)
        final_types = [parse_type_output(rt) for rt in raw_types]

        # 3. Batch Descriptions
        desc_prompts = []
        for i, item in enumerate(batch_data):
            tag_text = ", ".join(tags_list[i]) if tags_list[i] else "None"
            desc_prompts.append(DESC_PROMPT.format(mention_name=item['mention_name'], mention_type=final_types[i], mention_context=item['original_row'].get('sentence',''), image_tags=tag_text))
            
        raw_descs = self._generate_text_batched(desc_prompts, max_tokens=120)
        final_descs = [clean_description(parse_desc_output(rd), batch_data[i]['mention_name']) for i, rd in enumerate(raw_descs)]

        # 4. Batch Embeddings
        final_texts = []
        for i, item in enumerate(batch_data):
            tag_text = ", ".join(tags_list[i]) if tags_list[i] else "None"
            final_texts.append(f"Sentence: {item['sentence']} | Image Tags: {tag_text} | Entity: {item['mention_name']} ({final_types[i]}) - {final_descs[i]}")

        with torch.no_grad():
            emb_tensors = self.embed_model.encode(final_texts, convert_to_tensor=True)
            embeddings_list = emb_tensors.tolist()

        del emb_tensors
        torch.cuda.empty_cache()
        gc.collect()

        # 5. Return Results
        results = []
        for i in range(len(batch_data)):
            results.append({
                "tags": tags_list[i],
                "type": final_types[i],
                "description": final_descs[i],
                "concatenated_text": final_texts[i],
                "embedding": embeddings_list[i]
            })
        return results

# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================

def run_pipeline(pipe=None):
    if pipe is None:
        pipe = EntityExtractionPipeline()

    # pipe = EntityExtractionPipeline()
    
    dataset_path = CONFIG["DATASET_FILE"]
    #output_path = CONFIG["OUTPUT_FILE"]
    
    processed_meta = []
    processed_embeddings = []
    processed_indices = set()
    
    meta_path = CONFIG["OUTPUT_FILE_META"]
    emb_path = CONFIG["OUTPUT_FILE_EMB"]

    if os.path.exists(meta_path):
        try:
            # 1. Load the metadata
            with open(meta_path, 'r', encoding='utf-8') as f:
                processed_meta = json.load(f)
            
            # 2. Load the embeddings (THIS WAS MISSING)
            if os.path.exists(emb_path):
                with open(emb_path, 'r', encoding='utf-8') as f:
                    processed_embeddings = json.load(f)
            else:
                processed_embeddings = [] # Or {} depending on your format

            # 3. Get IDs from both files
            meta_ids = {item['id'] for item in processed_meta}
            
            # Handle embeddings IDs (assumes it's a list of dicts with 'id', 
            # change to `set(processed_embeddings.keys())` if it's a dictionary)
            if isinstance(processed_embeddings, dict):
                emb_ids = set(processed_embeddings.keys())
            else:
                emb_ids = {item['id'] for item in processed_embeddings if 'id' in item}

            # 4. Only consider an item "processed" if it is in BOTH files
            processed_indices = meta_ids.intersection(emb_ids)

            # 5. Trim both lists to remove any incomplete orphans
            processed_meta = [m for m in processed_meta if m['id'] in processed_indices]
            
            if isinstance(processed_embeddings, list):
                processed_embeddings = [e for e in processed_embeddings if e.get('id') in processed_indices]
            elif isinstance(processed_embeddings, dict):
                processed_embeddings = {k: v for k, v in processed_embeddings.items() if k in processed_indices}

            logger.info(f"Resuming from checkpoint. {len(processed_indices)} fully processed in both files.")
            
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}. Starting fresh.")

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)
        logger.info(f"Successfully loaded {len(full_dataset)} items from JSON.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)

    remaining_dataset = [row for row in full_dataset if row.get("id") not in processed_indices]
    logger.info(f"Items remaining to process: {len(remaining_dataset)}")

    for i in range(0, len(remaining_dataset), BATCH_SIZE):
        chunk = remaining_dataset[i : i + BATCH_SIZE]
        
        batch_requests = []
        for row in chunk:
            #original_img_name = row.get("image_list", "")
            #fixed_img_name = original_img_name.replace("mention_", "train_")
            #absolute_img_path = os.path.join(CONFIG["IMAGE_DIR"], fixed_img_name)
            
            """
            if original_img_name.strip() == "":
                absolute_img_path = ""
            else:
                absolute_img_path = os.path.join(CONFIG["IMAGE_DIR"], original_img_name)

            batch_requests.append({
                "id": row.get("id"),
                "sentence": row.get("sentence", ""),
                "mention_name": row.get("mentions", ""),
                "img_path": absolute_img_path,
                "original_row": row
            })
            """
            original_img_name = row.get("imgPath", "")
            
            if original_img_name.strip() == "":
                absolute_img_path = ""
            else:
                absolute_img_path = os.path.join(CONFIG["IMAGE_DIR"], original_img_name)

            # Mentions use "id", "sentence", and "mentions"
            batch_requests.append({
                "id": row.get("id"),
                "sentence": row.get("sentence", ""), 
                "mention_name": row.get("mentions", ""),
                "img_path": absolute_img_path,
                "original_row": row
            })
            
        logger.info(f"\nProcessing Batch {i // BATCH_SIZE + 1} (IDs: {[r['id'] for r in batch_requests]})...")
        
        try:
            batch_results = pipe.process_batch(batch_requests)
            
            for j, result_dict in enumerate(batch_results):
                logger.info(f"Generated text: {result_dict['concatenated_text']}")
                
                row = batch_requests[j]["original_row"].copy()

# ---------- META JSON ----------
                meta_entry = row.copy()

                meta_entry["generated_tags"] = result_dict["tags"]
                meta_entry["generated_type"] = result_dict["type"]
                meta_entry["generated_description"] = result_dict["description"]
                meta_entry["concatenated_text"] = result_dict["concatenated_text"]

# remove unwanted fields
                meta_entry.pop("embedding", None)
                meta_entry.pop("cands", None)

                processed_meta.append(meta_entry)


# ---------- EMBEDDINGS JSON ----------
                """
                emb_entry = {
                "id": row.get("id"),
                "qid": row.get("answer"),
                "name": row.get("entities"),
                "embedding": result_dict["embedding"]
                }  
                """
                emb_entry = {
                "id": batch_requests[j]["id"],
                "qid": row.get("answer"),       # Richpedia mentions use 'answer'
                "name": row.get("entities"),    # Richpedia mentions use 'entities'
                "embedding": result_dict["embedding"]
                }

                processed_embeddings.append(emb_entry)

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(processed_meta, f)

            with open(emb_path, 'w', encoding='utf-8') as f:
                json.dump(processed_embeddings, f)
                
        except Exception as e:
            logger.error(f"Failed processing Batch. Error: {e}")
            
    logger.info("\nDataset processing complete!")

if __name__ == "__main__":
    # initialize CONFIG first
    CONFIG = {**BASE_CONFIG, **SPLITS[0]}

    # Load models ONCE
    pipe = EntityExtractionPipeline()
    
    for split_cfg in SPLITS:
        CONFIG = {**BASE_CONFIG, **split_cfg}
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing split: {split_cfg['DATASET_FILE'].split('/')[-1]}")
        logger.info(f"{'='*50}")
        run_pipeline(pipe)  # pass pipe in
