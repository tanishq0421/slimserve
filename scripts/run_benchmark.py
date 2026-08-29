"""Thin CLI entrypoint: config file -> one benchmark row.

Usage:
    python -m scripts.run_benchmark --config configs/teacher_fp16.yaml
"""
from __future__ import annotations

import argparse

import yaml

from slimserve.core.config import engine_config_from_dict
from slimserve.pipeline import benchmark_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    engine_cfg = engine_config_from_dict(cfg)   # TP=2 spans both T4s for FP16 7B
    result = benchmark_config(
        config_name=cfg["config_name"],
        engine_cfg=engine_cfg,
        params_b=cfg["params_b"],
        gpu_hourly_usd=cfg["gpu_hourly_usd"],
        evaluator_name=cfg.get("evaluator"),          # e.g. "toolcall" for real accuracy
        evaluator_samples=cfg.get("eval_samples", 200),
    )
    print(result)


if __name__ == "__main__":
    main()
