from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

import tiny_llama_v1 as tlm

sys.modules[__name__].TinyLlamaConfig = tlm.TinyLlamaConfig



def chat(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    delay=0.03,
):
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

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


def _load_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _build_tokenizer(tokenizer_repo: str) -> PreTrainedTokenizerFast:
    tokenizer_path = hf_hub_download(tokenizer_repo, "tokenizer.json")

    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )



def load_model_and_tokenizer(
    *,
    hf_repo: str | None,
    tokenizer_repo: str | None,
    config_path: Path | None,
    weights_path: Path | None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer_repo is None:
        raise ValueError("tokenizer_repo is required")

    tokenizer = _build_tokenizer(tokenizer_repo)

    if config_path is not None and config_path.is_file():
        cfg = tlm.load_tiny_llama_config(config_path)

    elif hf_repo:
        hub_cfg = hf_hub_download(hf_repo, "config.json")
        with open(hub_cfg) as f:
            cfg_dict = json.load(f)

        cfg_dict.pop("model_type", None)
        cfg = tlm.TinyLlamaConfig(**cfg_dict)

    else:
        raise ValueError("Need config")

    cfg.device = str(device)

    model = tlm.TinyLlamaForCausalLM(cfg)

    if weights_path is not None:
        wpath = weights_path
    elif hf_repo:
        wpath = Path(hf_hub_download(hf_repo, "best_model.pt"))
    else:
        raise ValueError("Need weights")

    ckpt = torch.load(wpath, map_location="cpu", weights_only=False)

    model.load_state_dict(_load_state_dict(ckpt))
    model.eval()

    model = model.to(device)

    print(f"Loaded on {device}")

    return model, tokenizer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--repo", default=None)
    p.add_argument("--tokenizer-repo", default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None)

    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--delay", type=float, default=0.02)

    return p.parse_args()

def main():
    args = parse_args()

    hf_repo = args.repo
    tokenizer_repo = args.tokenizer_repo or hf_repo

    if hf_repo is None:
        hf_repo = input("HF repo: ").strip()

    if tokenizer_repo is None:
        tokenizer_repo = hf_repo

    model, tokenizer = load_model_and_tokenizer(
        hf_repo=hf_repo,
        tokenizer_repo=tokenizer_repo,
        config_path=args.config,
        weights_path=args.weights,
    )

    prompt = input("Prompt: ").strip() or "<s>"
    print("output: ")
    chat(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()