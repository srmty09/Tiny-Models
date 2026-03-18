import tiny_llama_v1
import sys
import torch
import json
import time as _time
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

sys.modules[__name__].TinyLlamaConfig = tiny_llama_v1.TinyLlamaConfig


def chat(model, tokenizer, prompt, max_new_tokens=100,
         temperature=0.8, top_k=50, top_p=0.9, delay=0.03):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] >= model.model.cfg.max_position_embedding:
        input_ids = input_ids[:, -model.model.cfg.max_position_embedding + max_new_tokens:]
        print(f"(prompt truncated to {input_ids.shape[1]} tokens)")
    print(f"\033[1m{prompt}\033[0m", end=" ", flush=True)
    for token_text in model.generate(
        input_ids,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    ):
        print(token_text, end="", flush=True)
        _time.sleep(delay)
    print()

repo_id = "srmty/tiny-llama-71m"

tokenizer_path = hf_hub_download(repo_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_path,
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>"
)

config_path = hf_hub_download(repo_id, "config.json")
with open(config_path) as f:
    cfg_dict = json.load(f)
cfg_dict.pop("model_type", None)
cfg = tiny_llama_v1.TinyLlamaConfig(**cfg_dict)

model = tiny_llama_v1.TinyLlamaForCausalLM(cfg)

weights_path = hf_hub_download(repo_id, "latest.pt")
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

if isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

clean_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("_orig_mod.", "")
    clean_state_dict[new_key] = v

model.load_state_dict(clean_state_dict)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

chat(model, tokenizer, "A old man is sitting")