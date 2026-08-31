"""Quantize a model via a registered Quantizer (AWQ), then serve it with vLLM.

    python -m scripts.quantize --method awq --precision int4 \
        --model tanishq0421/slimserve-student_1p5b_gold \
        --out   checkpoints/student_1p5b_gold_awq

The output dir is what ``run_benchmark --model <out>`` serves (config sets
``quantization: awq``). Run this in its own step — AutoAWQ pulls its own deps and
is best kept separate from the vLLM serving environment.
"""
from __future__ import annotations

import argparse

import slimserve.quantization.awq  # noqa: F401  side-effect: register "awq"
from slimserve.core.config import Precision, QuantConfig
from slimserve.core.registry import build


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="awq")
    p.add_argument("--model", required=True, help="fp16 checkpoint (local dir or HF id)")
    p.add_argument("--out", required=True, help="where to write the quantized model")
    p.add_argument("--precision", default="int4", choices=["int4", "int8"])
    args = p.parse_args()

    quantizer = build("quantizer", args.method)
    cfg = QuantConfig(name=args.method, precision=Precision(args.precision),
                      out_path=args.out)
    out = quantizer.quantize(args.model, cfg)
    print(f"quantized {args.model} -> {out}")


if __name__ == "__main__":
    main()
