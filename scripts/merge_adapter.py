"""Merge a trained LoRA adapter into its base → a standalone checkpoint vLLM can serve.

Run these ONE AT A TIME. It reloads the fp16 base on CPU (~3 GB RAM for 1.5B);
running several at once is what OOM'd the in-training merge.

    python -m scripts.merge_adapter \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --adapter checkpoints/student_1p5b_gold_adapter \
        --out     checkpoints/student_1p5b_gold
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base model id (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    p.add_argument("--adapter", required=True, help="path to the saved LoRA adapter dir")
    p.add_argument("--out", required=True, help="where to write the merged standalone model")
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(       # CPU load, avoids GPU OOM
        args.base, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"merged {args.adapter} -> {args.out}")


if __name__ == "__main__":
    main()
