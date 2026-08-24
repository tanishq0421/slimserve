"""Benchmark result record + collection helpers.

BenchmarkResult is the single row that gets appended to results/benchmarks.csv.
It is the source of truth for the whole project's story.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    config_name: str          # e.g. "student_distilled_int4"
    params_b: float           # billions of parameters
    precision: str            # fp16 | int8 | int4 | fp8
    # quality
    tool_acc: float           # BFCL tool-selection accuracy
    arg_acc: float            # argument correctness
    # performance
    tokens_per_s: float
    ttft_ms: float
    p99_latency_ms: float
    vram_mb: float
    # cost
    cost_per_1m_tokens: float

    def as_row(self) -> dict:
        return asdict(self)


def cost_per_1m_tokens(gpu_hourly_usd: float, tokens_per_s: float) -> float:
    """$/1M tokens = GPU $/hr / tokens-per-hour. State the reference GPU + rate."""
    tokens_per_hour = tokens_per_s * 3600.0
    return gpu_hourly_usd / (tokens_per_hour / 1_000_000)
