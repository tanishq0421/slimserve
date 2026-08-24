"""BenchmarkRunner — orchestrates a load test + eval for one config.

Depends only on the InferenceEngine and Evaluator *interfaces* (DIP), so it runs
identically against vLLM, HF, or the from-scratch engine. This is the class that
turns "swappable strategies" into one comparable benchmark row.
"""
from __future__ import annotations

from slimserve.core.interfaces import Evaluator, InferenceEngine
from slimserve.evaluation.metrics import BenchmarkResult, cost_per_1m_tokens


class BenchmarkRunner:
    def __init__(self, gpu_hourly_usd: float) -> None:
        self.gpu_hourly_usd = gpu_hourly_usd

    def run(
        self,
        config_name: str,
        engine: InferenceEngine,
        evaluator: Evaluator,
        params_b: float,
        precision: str,
    ) -> BenchmarkResult:
        # 1. load-test the engine -> throughput/latency/vram
        # 2. evaluator.evaluate(engine) -> quality
        # 3. cost_per_1m_tokens(self.gpu_hourly_usd, tokens_per_s)
        raise NotImplementedError(
            "Phase 1 Wk1: load-test loop -> assemble BenchmarkResult."
        )
