# Tiny Llama v1

A 72M parameter Llama-style causal language model built from scratch in PyTorch. Features Grouped Query Attention (GQA), RoPE, RMSNorm, SwiGLU MLP, and KV-cache streaming generation.

Pre-trained weights: [srmty/tiny-llama-71M](https://huggingface.co/srmty/tiny-llama-71M/)

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~72M |
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| KV heads | 4 (GQA) |
| Head dim | 64 |
| FFN hidden dim | 1024 |
| Max sequence length | 256 |
| Vocab size | 10,000 |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Activation | SwiGLU |

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/srmty09/Tiny-Models.git
cd Tiny-Models/Llama
```

### 2. Install dependencies

```bash
pip install torch transformers huggingface_hub
```

### 3. Download the model checkpoint

Download `latest.pt` from [srmty/tiny-llama-71M](https://huggingface.co/srmty/tiny-llama-71M/) and place it in the `Llama/` directory.

### 4. Run inference

```bash
python run_model.py
```

Enter a prompt when asked (e.g. "Once upon a time") and the model will stream generated text to the terminal.

## Project Structure

```
Llama/
├── tiny_llama_v1.py   # Model architecture and config
├── run_model.py       # Interactive inference script
└── README.md
```

## License

MIT
