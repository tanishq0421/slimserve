"""End-to-end orchestration: compress -> serve -> evaluate -> record.

Wires the stages together using only the registry + interfaces. Importing this
module also registers every implementation (the imports below trigger the
``@register`` decorators), so ``build(...)`` can find them by name.
"""
from __future__ import annotations

# --- side-effect imports: populate the registry -----------------------------
from slimserve.engines import vllm_engine          # noqa: F401
from slimserve.quantization import awq             # noqa: F401
from slimserve.training import qlora_trainer       # noqa: F401
from slimserve.training.distillation import logit_kd, sequence_kd  # noqa: F401
from slimserve.evaluation import bfcl_evaluator    # noqa: F401
# ----------------------------------------------------------------------------

from slimserve.benchmark.runner import BenchmarkRunner
from slimserve.benchmark.results_store import ResultsStore
from slimserve.core.config import EngineConfig
from slimserve.core.registry import build


def benchmark_config(
    config_name: str,
    engine_cfg: EngineConfig,
    params_b: float,
    gpu_hourly_usd: float,
    evaluator_name: str = "bfcl",
) -> None:
    """Run one config and append its row to the hero table."""
    engine = build("engine", engine_cfg.name, engine_cfg)
    evaluator = build("evaluator", evaluator_name)
    runner = BenchmarkRunner(gpu_hourly_usd=gpu_hourly_usd)

    result = runner.run(
        config_name=config_name,
        engine=engine,
        evaluator=evaluator,
        params_b=params_b,
        precision=engine_cfg.precision.value,
    )
    ResultsStore().append(result)
