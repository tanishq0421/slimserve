"""Thin CLI entrypoint: config file -> one benchmark row.

Usage:
    python -m scripts.run_benchmark --config configs/teacher_fp16.yaml
"""
from __future__ import annotations

import argparse

import yaml

from slimserve.core.config import EngineConfig, Precision
from slimserve.pipeline import benchmark_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    engine_cfg = EngineConfig(
        name=cfg["engine"],
        model_path=cfg["model_path"],
        precision=Precision(cfg.get("precision", "fp16")),
        kv_cache_quant=cfg.get("kv_cache_quant", False),
        max_num_seqs=cfg.get("max_num_seqs", 256),
    )
    benchmark_config(
        config_name=cfg["config_name"],
        engine_cfg=engine_cfg,
        params_b=cfg["params_b"],
        gpu_hourly_usd=cfg["gpu_hourly_usd"],
    )


if __name__ == "__main__":
    main()
