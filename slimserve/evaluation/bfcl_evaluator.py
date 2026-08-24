"""Berkeley Function-Calling Leaderboard evaluator (Phase 1 onward).

Scores an engine on tool-calling: did it pick the right tool, with valid,
correct arguments? Exact metrics — no fuzzy human grading.
"""
from __future__ import annotations

from slimserve.core.interfaces import Evaluator, InferenceEngine
from slimserve.core.registry import register


@register("evaluator", "bfcl")
class BFCLEvaluator(Evaluator):
    def __init__(self, split: str = "simple") -> None:
        self.split = split

    def evaluate(self, engine: InferenceEngine) -> dict[str, float]:
        # for each sample: build GenerationRequest with tool schemas,
        # engine.generate(), parse the tool call, compare to gold.
        raise NotImplementedError("Phase 1 Wk2+: wire BFCL harness.")
