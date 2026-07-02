# VSCE-EL: Visual-Semantic Cross-Encoder Entity Linking

Official implementation of the VSCE-EL pipeline for Multimodal Entity Linking.

## Results

| Model | Rich @1 | WM @1 | WD @1 |
|---|---|---|---|
| Hybrid Retrieval | 68.13 | 93.06 | 72.30 |
| BGE-Only | 89.46 | 87.34 | 86.25 |
| w/o Fine Tuning | 87.94 | 86.63 | 86.50 |
| VSCE-EL w/o Caption | 92.67 | 94.26 | 89.74 |
| **VSCE-EL (Ours)** | **93.38** | **94.75** | **93.12** |

## Pipeline

Stage 0  →  Mention Enhancement (RAM++ tags, Qwen2 type+desc, LLaVA captions)

Stage 1  →  Entity Representation (LLAMA summaries + SFR-Mistral embeddings)

Stage 2  →  Hybrid Retrieval (fuzzy 0.2 + cosine 0.8 → Top-50)

Stage 3  →  Cross-Encoder Reranking (BGE fine-tuned → Top-10)

Stage 4  →  LLM Reranking (Qwen2-7B + LoRA → Final prediction)


## Setup

```bash
conda create -n vsce_el python=3.9
conda activate vsce_el
pip install -r requirements.txt
```

## Datasets

Download and place under `data/`:
- RichpediaMEL + WikiMEL: https://github.com/seukgcode/MELBench
- WikiDiverse: https://github.com/wangxw5/wikiDiverse

## Models

Download and place under `models/`:
| Model | HuggingFace Link |
|---|---|
| Qwen2-7B-Instruct | https://huggingface.co/Qwen/Qwen2-7B-Instruct |
| Meta-Llama-3-8B-Instruct | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct |
| LLaVA-v1.6-Mistral-7B | https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf |
| SFR-Embedding-Mistral | https://huggingface.co/Salesforce/SFR-Embedding-Mistral |
| RAM++ | https://github.com/xinyu1205/recognize-anything |

## Run

```bash
# Full pipeline
bash run.sh richpedia
bash run.sh wikimel
bash run.sh wikidiverse

# Individual stages
bash run.sh richpedia 0      # mention enhancement
bash run.sh richpedia 1      # entity embedding
bash run.sh richpedia 2      # hybrid retrieval
bash run.sh richpedia 3      # cross-encoder
bash run.sh richpedia 4      # LLM reranker
bash run.sh richpedia eval   # evaluation only
```
