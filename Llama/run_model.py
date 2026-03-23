"""
Interactive inference for Tiny Llama **v1** checkpoints (dense LM in ``tiny_llama_v1.py``).

No default Hugging Face repo: pass ``--repo`` or enter it when prompted.
For fully local weights + config, pass ``--weights``, ``--config``, and ``--tokenizer-repo``.

Examples:
  python run_model.py
  python run_model.py --repo srmty/tiny-llama-71m
  python run_model.py --weights ./latest.pt --config configs/tiny_llama_v1_base.json --tokenizer-repo srmty/tiny-llama-71m
"""
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

# Register config class on this module (helps if checkpoints reference it by name).
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
) -> tuple[tlm.TinyLlamaForCausalLM, PreTrainedTokenizerFast]:
    tok_repo = tokenizer_repo
    if tok_repo is None:
        raise ValueError("tokenizer_repo is required (same as model repo or --tokenizer-repo).")

    tokenizer = _build_tokenizer(tok_repo)

    if config_path is not None and config_path.is_file():
        cfg = tlm.load_tiny_llama_config(config_path)
    elif hf_repo:
        hub_cfg = hf_hub_download(hf_repo, "config.json")
        with open(hub_cfg) as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("model_type", None)
        cfg = tlm.TinyLlamaConfig(**cfg_dict)
    else:
        raise ValueError("Provide --config (local JSON) or a Hugging Face --repo with config.json.")

    model = tlm.TinyLlamaForCausalLM(cfg)

    if weights_path is not None:
        wpath = weights_path
    elif hf_repo:
        wpath = Path(hf_hub_download(hf_repo, "latest.pt"))
    else:
        raise ValueError("Provide --weights (local .pt) or a Hugging Face --repo with latest.pt.")

    ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
    model.load_state_dict(_load_state_dict(ckpt))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Tiny Llama v1 interactive generation. No default Hub repo — use prompts or flags."
    )
    p.add_argument(
        "--repo",
        default=None,
        metavar="ORG/NAME",
        help="Hugging Face repo with tokenizer.json, config.json, and latest.pt (optional if you use prompts).",
    )
    p.add_argument(
        "--tokenizer-repo",
        default=None,
        metavar="ORG/NAME",
        help="Repo that hosts tokenizer.json if different from --repo (required for local weights if not same).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Local JSON config (skips Hub config.json).",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Local checkpoint .pt (skips Hub latest.pt).",
    )
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--delay", type=float, default=0.03)
    return p.parse_args()


def _prompt_repo(label: str) -> str:
    s = input(label).strip()
    return s


def main():
    args = parse_args()
    hf_repo = args.repo
    tokenizer_repo = args.tokenizer_repo
    config_path = args.config
    weights_path = args.weights

    local_weights = weights_path is not None
    local_config = config_path is not None and config_path.is_file()

    # Fully local checkpoint: need tokenizer HF repo, no model repo required for files
    if local_weights and local_config:
        if tokenizer_repo is None:
            if hf_repo is not None:
                tokenizer_repo = hf_repo
            else:
                tokenizer_repo = _prompt_repo(
                    "Hugging Face repo id for tokenizer.json (weights/config are local): "
                )
        if not tokenizer_repo:
            print("Error: need --tokenizer-repo or --repo for tokenizer.json.", file=sys.stderr)
            sys.exit(1)
        # hf_repo only used for hub config/weights; may be None here
    else:
        # Need Hub model repo (or user will fail on missing weights/config)
        if hf_repo is None:
            hf_repo = _prompt_repo(
                "Hugging Face model repo id (tokenizer.json, config.json, latest.pt): "
            )
        if not hf_repo:
            print(
                "Error: need a model repo, or pass both --weights and --config with --tokenizer-repo.",
                file=sys.stderr,
            )
            sys.exit(1)
        if tokenizer_repo is None:
            tokenizer_repo = hf_repo

    if not tokenizer_repo:
        print("Error: tokenizer repo is required.", file=sys.stderr)
        sys.exit(1)

    try:
        model, tokenizer = load_model_and_tokenizer(
            hf_repo=hf_repo,
            tokenizer_repo=tokenizer_repo,
            config_path=config_path,
            weights_path=weights_path,
        )
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    initial_prompts = input("Give some initial lines (e.g. Once upon a time): ")
    if len(initial_prompts) == 0:
        initial_prompts = "<s>"
    chat(
        model,
        tokenizer,
        initial_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
