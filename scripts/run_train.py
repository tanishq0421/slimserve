"""QLoRA-fine-tune a student from a training YAML.

    python -m scripts.run_train --config configs/train_1p5b_gold.yaml

Pin to one GPU (so you can run two trainings at once, one per T4):

    CUDA_VISIBLE_DEVICES=0 python -m scripts.run_train --config configs/train_1p5b_gold.yaml
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_train --config configs/train_1p5b_distill.yaml
"""
from __future__ import annotations

import argparse

import yaml

import slimserve.training.qlora_trainer  # noqa: F401  side-effect: register "qlora"
from slimserve.core.config import train_config_from_dict
from slimserve.core.registry import build


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = build("trainer", cfg.get("trainer", "qlora"))
    out = trainer.train(train_config_from_dict(cfg))
    print(f"saved merged student to {out}")


if __name__ == "__main__":
    main()
