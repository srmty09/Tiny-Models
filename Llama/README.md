# Tiny Llama

Llama-style causal language models built from scratch in PyTorch: **Grouped Query Attention (GQA)**, **RoPE**, **RMSNorm**, **SwiGLU MLP**, and **KV-cache** streaming generation.

| Variant | ~Params | Config file | Weights (HF) | Training data |
|--------|---------|-------------|--------------|---------------|
| **v1 Small** | ~72M | `configs/tiny_llama_v1_small.json` | [srmty/tiny-llama-71M](https://huggingface.co/srmty/tiny-llama-71M/) | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~450M tokens) |
| **v1 Base** | ~128.49M | `configs/tiny_llama_v1_base.json` | [srmty/tiny-llama-128m](https://huggingface.co/srmty/tiny-llama-128m) (set repo if different) | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~0.6B tokens) |

**Dense v1 implementation:** `tiny_llama_v1.py` (architecture + `TinyLlamaConfig` + `load_tiny_llama_config()`). JSON files under `configs/` define **v1 Small** and **v1 Base** presets.

**MoE / v2:** `tiny_llama_v2.py` is a separate experimental stack (not wired into `run_model.py` yet).

## Architecture (v1 family)

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| KV heads | 4 (GQA) |
| Head dim | 64 |
| FFN hidden dim | 1024 (v1 Small) / **3072 (v1 Base)** |
| Max sequence length | 256 (v1 Small) / **512 (v1 Base)** |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Activation | SwiGLU |

See each JSON under `configs/` for exact training hyperparameters.

## How to run

### 1. Clone and install

```bash
git clone https://github.com/srmty09/Tiny-Models.git
cd Tiny-Models/Llama
pip install torch transformers huggingface_hub
```

### 2. Interactive inference (`run_model.py`)

There is **no default** Hugging Face repository. Either pass `--repo` or type the repo id when prompted.

**Everything on the Hub (tokenizer, config, weights):**

```bash
python run_model.py
# Enter e.g. srmty/tiny-llama-71m when asked

# Or non-interactive:
python run_model.py --repo srmty/tiny-llama-71m
```

**Local checkpoint + local config** (tokenizer still downloaded from Hub), e.g. **v1 Base**:

```bash
python run_model.py \
  --weights ./latest.pt \
  --config configs/tiny_llama_v1_base.json \
  --tokenizer-repo srmty/tiny-llama-71m
```

**Local weights but Hub config** (same repo for config + tokenizer):

```bash
python run_model.py --repo your-org/your-repo --weights ./latest.pt
```

Optional: `--tokenizer-repo` if `tokenizer.json` lives on a different repo than the model files.

### 3. Load config in Python

```python
from tiny_llama_v1 import load_tiny_llama_config, TinyLlamaForCausalLM

cfg = load_tiny_llama_config("configs/tiny_llama_v1_base.json")
model = TinyLlamaForCausalLM(cfg)
```

## Project layout

```
Llama/
├── tiny_llama_v1.py            # Dense v1 model + config loader
├── tiny_llama_v2.py            # MoE / v2 experiment (separate)
├── configs/
│   ├── tiny_llama_v1_small.json   # v1 Small / ~72M
│   └── tiny_llama_v1_base.json    # v1 Base / ~128M
├── run_model.py                # CLI inference (prompts for HF repo if omitted)
└── README.md
```

## Website (GitHub Pages)

The static site is in the repo’s **`docs/`** folder (sibling of `Llama/`):

- **`index.html`** — model cards (home)
- **`tiny-llama-v1-small.html`** — v1 Small / TinyStories
- **`tiny-llama-v1-base.html`** — v1 Base / FineWeb-Edu
- **`site.css`** — shared styles
- **`assets/`** — images (`tiny-llama-v1-small-training.png`, `tiny-llama-v1-base-training.png`)

## License

MIT
