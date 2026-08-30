"""End-to-end orchestration: compress -> serve -> evaluate -> record.

Wires the stages together using only the registry + interfaces. Importing this
module also registers every implementation (the imports below trigger the
``@register`` decorators), so ``build(...)`` can find them by name.
"""
from __future__ import annotations

# --- side-effect imports: populate the registry -----------------------------
from slimserve.engines import vllm_engine          # noqa: F401
from slimserve.engines.mini_engine import engine as mini_engine  # noqa: F401
from slimserve.quantization import awq             # noqa: F401
from slimserve.training import qlora_trainer       # noqa: F401
from slimserve.training.distillation import logit_kd, sequence_kd  # noqa: F401
from slimserve.evaluation import toolcall_evaluator  # noqa: F401
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
    evaluator_name: str | None = None,
    evaluator_samples: int = 200,
) -> "BenchmarkResult":
    """Run one config and append its row to the hero table.

    ``evaluator_name`` is None to use the built-in well-formedness stand-in, or
    "toolcall" for real tool-calling accuracy on the xLAM held-out set.
    """
    engine = build("engine", engine_cfg.name, engine_cfg)
    evaluator = (
        build("evaluator", evaluator_name, num_samples=evaluator_samples)
        if evaluator_name
        else None
    )
    runner = BenchmarkRunner(gpu_hourly_usd=gpu_hourly_usd)

    result = runner.run(
        config_name=config_name,
        engine=engine,
        params_b=params_b,
        precision=engine_cfg.precision.value,
        evaluator=evaluator,
    )
    ResultsStore().append(result)
    return result
