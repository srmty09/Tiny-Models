# Tiny Llama

Llama-style causal LMs in PyTorch: **GQA**, **RoPE**, **RMSNorm**, **SwiGLU MLP**, and **KV-cache** streaming generation.

| Variant | ~Params | Config | Weights (HF) | Data |
|--------|---------|--------|--------------|------|
| **v1 Small** | ~72M | `configs/tiny_llama_v1_small.json` | [srmty/tiny-llama-71M](https://huggingface.co/srmty/tiny-llama-71M) | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~450M tokens) |
| **v1 Base** | ~128M | `configs/tiny_llama_v1_base.json` | [srmty/tiny-llama-128m](https://huggingface.co/srmty/tiny-llama-128m) | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~0.6B tokens in the original recipe; longer runs below) |

**Code:** `tiny_llama_v1.py` — `TinyLlamaConfig`, `load_tiny_llama_config()`, model classes. **`tiny_llama_v2.py`** is a separate MoE experiment (not used by `run_model.py`).

## Architecture (v1)

| Parameter | v1 Small | v1 Base |
|-----------|----------|---------|
| Layers | 12 | 12 |
| Hidden dim | 768 | 768 |
| Heads / KV heads | 12 / 4 | 12 / 4 |
| Head dim | 64 | 64 |
| FFN dim | 1024 | 3072 |
| Max seq len | 256 | 512 |

RMSNorm, RoPE, SwiGLU. Full training hyperparameters are in each JSON under `configs/`.

## Training notes (v1 Base, FineWeb-Edu)

From a Weights & Biases run on **FineWeb-Edu** (logged as two contiguous runs after a checkpoint resume around **~12–12.5k steps**):

| Metric | Approximate value |
|--------|-------------------|
| Steps | ~25–26k |
| Tokens seen | ~**1.1B** |
| Peak learning rate | **3×10⁻⁴** (linear warmup in the first ~2k steps, then decay) |
| Throughput | ~**15.5k** tokens/s before resume, ~**12.5k** after (noisy; depends on hardware and dataloader) |
| **Validation loss** (end) | ~**3.1** |
| **Validation perplexity** (end) | ~**20–25** |

Train loss falls quickly early, then stays in the **~3.5** range in the stable phase; **loss per token** and throughput can jump when resuming from checkpoint (new process, cache state, or data shard). See the project’s W&B project (`wandb_project` in the JSON configs) for raw curves.

## Run inference (`run_model.py`)

Install: `pip install torch transformers huggingface_hub`

`run_model.py` downloads **`config.json`**, **`tokenizer.json`**, and **`best_model.pt`** from the Hub unless you override paths. Pass **`--repo`** or enter the repo id when prompted. Optional **`--tokenizer-repo`** if the tokenizer lives elsewhere.

```bash
cd Tiny-Models/Llama

python run_model.py --repo srmty/tiny-llama-71m
```

**Local weights** (tokenizer still from Hub):

```bash
python run_model.py \
  --weights ./best_model.pt \
  --config configs/tiny_llama_v1_base.json \
  --tokenizer-repo srmty/tiny-llama-128m
```

**Hub config + local weights only:**

```bash
python run_model.py --repo your-org/your-repo --weights ./best_model.pt
```

## Load config in code

```python
from tiny_llama_v1 import load_tiny_llama_config, TinyLlamaForCausalLM

cfg = load_tiny_llama_config("configs/tiny_llama_v1_base.json")
model = TinyLlamaForCausalLM(cfg)
```

## Layout

```
Llama/
├── tiny_llama_v1.py
├── tiny_llama_v2.py
├── configs/
│   ├── tiny_llama_v1_small.json
│   └── tiny_llama_v1_base.json
├── run_model.py
└── README.md
```

Static pages for GitHub Pages live under repo root **`docs/`** (`index.html`, model pages, `site.css`).

## License

MIT
