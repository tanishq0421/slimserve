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
    parser.add_argument("--model", help="override model_path (e.g. a local checkpoint dir)")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if this config is already in the results store")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model:                              # serve a local dir instead of the HF repo
        cfg["model_path"] = args.model

    # Resume-friendly: skip configs already recorded (a killed suite re-runs and
    # continues; a single config re-run avoids a duplicate row). --force overrides.
    from slimserve.benchmark.results_store import ResultsStore
    if not args.force and ResultsStore().has(cfg["config_name"]):
        print(f"skip {cfg['config_name']}: already in results (use --force to re-run)")
        return

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
