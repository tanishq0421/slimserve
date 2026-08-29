"""Build Phase 3 training data from xLAM.

    # gold SFT set (instant — just formatting the ground-truth calls):
    python -m scripts.build_data --mode gold --num 10000 --out data/sft_gold.jsonl

    # teacher-distillation set (runs the teacher via vLLM to generate targets):
    python -m scripts.build_data --mode teacher \
        --teacher-config configs/teacher_int4_awq.yaml \
        --num 10000 --out data/teacher_distill.jsonl

The INT4 teacher is a fine generator: same tool accuracy as FP16, single T4, faster.
"""
from __future__ import annotations

import argparse

import yaml

from slimserve.data.build import build_gold, build_teacher


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gold", "teacher"], required=True)
    p.add_argument("--num", type=int, default=10000)
    p.add_argument("--out", required=True)
    p.add_argument("--teacher-config", help="engine YAML (teacher mode only)")
    p.add_argument("--keep-all", action="store_true",
                   help="teacher mode: keep even wrong-tool completions")
    p.add_argument("--tensor-parallel", type=int, default=None,
                   help="override TP for teacher gen (e.g. 2 to use both T4s)")
    args = p.parse_args()

    if args.mode == "gold":
        n = build_gold(args.num, args.out)
    else:
        if not args.teacher_config:
            p.error("--teacher-config is required for teacher mode")
        import slimserve.pipeline  # noqa: F401  side-effect: register engines
        from slimserve.core.config import engine_config_from_dict
        from slimserve.core.registry import build
        with open(args.teacher_config) as f:
            cfg = yaml.safe_load(f)
        if args.tensor_parallel:                    # e.g. 2 -> use both T4s
            cfg["tensor_parallel_size"] = args.tensor_parallel
        engine = build("engine", cfg["engine"], engine_config_from_dict(cfg))
        n = build_teacher(args.num, engine, args.out,
                          only_correct=not args.keep_all)

    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
