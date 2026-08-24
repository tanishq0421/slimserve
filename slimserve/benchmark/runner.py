"""BenchmarkRunner — orchestrates a load test + quality check for one config.

Depends only on the InferenceEngine (and optional Evaluator) *interface* (DIP),
so it runs identically against vLLM, HF, or the from-scratch engine. Produces one
comparable BenchmarkResult row.

Methodology (Phase 1):
  * Throughput  — send the whole workload as one batch; vLLM batches internally;
                  tok/s = total completion tokens / wall time.
  * Latency     — send requests one at a time (batch=1) and record each; report p99.
  * TTFT        — proxy = latency of a prefill-only (max_tokens=1) request. True
                  streaming TTFT arrives in Week 2 with the online server.
  * Quality     — Week 1: fraction of well-formed tool calls. Week 2: BFCLEvaluator.
"""
from __future__ import annotations

import time
from dataclasses import replace
from statistics import mean

from slimserve.benchmark.workload import build_tool_calling_workload
from slimserve.core.interfaces import Evaluator, InferenceEngine
from slimserve.evaluation.metrics import BenchmarkResult, cost_per_1m_tokens
from slimserve.evaluation.validity import is_valid_tool_call


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[k]


class BenchmarkRunner:
    def __init__(self, gpu_hourly_usd: float) -> None:
        self.gpu_hourly_usd = gpu_hourly_usd

    def run(
        self,
        config_name: str,
        engine: InferenceEngine,
        params_b: float,
        precision: str,
        evaluator: Evaluator | None = None,
        num_requests: int = 64,
        latency_samples: int = 16,
    ) -> BenchmarkResult:
        workload = build_tool_calling_workload(num_requests)

        # 1. warmup (compiles kernels / fills caches; excluded from timing)
        engine.generate(workload[0])

        # 2. throughput — one big batch
        t0 = time.perf_counter()
        outputs = engine.generate_batch(workload)
        elapsed_s = time.perf_counter() - t0
        total_completion = sum(o.completion_tokens for o in outputs)
        tokens_per_s = total_completion / elapsed_s if elapsed_s > 0 else 0.0

        # 3. latency — sequential, batch=1
        latencies = [engine.generate(r).latency_ms for r in workload[:latency_samples]]
        p99 = _percentile(latencies, 99)

        # 4. TTFT proxy — prefill only
        ttfts = [
            engine.generate(replace(r, max_tokens=1)).latency_ms
            for r in workload[:latency_samples]
        ]
        ttft_ms = mean(ttfts) if ttfts else 0.0

        # 5. quality
        if evaluator is not None:
            metrics = evaluator.evaluate(engine)
            tool_acc = metrics.get("tool_acc", 0.0)
            arg_acc = metrics.get("arg_acc", 0.0)
        else:
            valid_rate = mean(is_valid_tool_call(o.text) for o in outputs)
            tool_acc = arg_acc = valid_rate   # Week-1 stand-in

        # 6. memory + cost
        vram_mb = engine.memory_footprint().peak_mb
        cost = cost_per_1m_tokens(self.gpu_hourly_usd, tokens_per_s)

        return BenchmarkResult(
            config_name=config_name,
            params_b=params_b,
            precision=precision,
            tool_acc=round(tool_acc, 4),
            arg_acc=round(arg_acc, 4),
            tokens_per_s=round(tokens_per_s, 2),
            ttft_ms=round(ttft_ms, 2),
            p99_latency_ms=round(p99, 2),
            vram_mb=round(vram_mb, 1),
            cost_per_1m_tokens=round(cost, 4),
        )
