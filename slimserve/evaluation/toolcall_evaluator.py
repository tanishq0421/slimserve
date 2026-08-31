"""Tool-calling accuracy on a real dataset (Phase 1, Week 2 — the real quality metric).

Scores the model against a held-out slice of Salesforce xLAM-function-calling-60k:
for each (query, tools, gold call), does the model pick the right tool with the
right arguments? Replaces the Week-1 well-formedness stand-in.

Honest scope (see FINDINGS): single-tool-call examples only, strict exact-match on
arguments. It is a *relative* metric for comparing our own configs, not the official
BFCL leaderboard score.
"""
from __future__ import annotations

import json

from slimserve.core.config import GenerationRequest
from slimserve.core.interfaces import Evaluator, InferenceEngine
from slimserve.core.registry import register
from slimserve.evaluation.parse import args_match, parse_tool_call, xlam_tools_to_openai


@register("evaluator", "toolcall")
class ToolCallAccuracyEvaluator(Evaluator):
    def __init__(
        self,
        num_samples: int = 200,
        dataset: str = "Salesforce/xlam-function-calling-60k",
        split: str = "train",
        max_tokens: int = 256,
    ) -> None:
        self.num_samples = num_samples
        self.dataset = dataset
        self.split = split
        self.max_tokens = max_tokens

    def _load_examples(self) -> list[tuple[str, list, dict]]:
        """Held-out single-call examples, taken from the END of the dataset so the
        front stays free for Phase-2 training (no leakage)."""
        from datasets import load_dataset

        ds = load_dataset(self.dataset, split=self.split)
        examples: list[tuple[str, list, dict]] = []
        for i in range(len(ds) - 1, -1, -1):
            row = ds[i]
            try:
                tools = json.loads(row["tools"])
                answers = json.loads(row["answers"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if len(answers) != 1:                 # keep the metric clean: single call
                continue
            examples.append((row["query"], tools, answers[0]))
            if len(examples) >= self.num_samples:
                break
        return examples

    def evaluate(self, engine: InferenceEngine) -> dict[str, float]:
        examples = self._load_examples()
        if not examples:
            return {"tool_acc": 0.0, "arg_acc": 0.0}

        requests = [
            GenerationRequest(
                prompt=query,
                tools=tuple(xlam_tools_to_openai(tools)),
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            for query, tools, _gold in examples
        ]
        outputs = engine.generate_batch(requests)   # batched for speed

        tool_ok = 0
        full_ok = 0
        for (_q, _t, gold), out in zip(examples, outputs):
            pred = parse_tool_call(out.text)
            if pred and pred["name"] == gold.get("name"):
                tool_ok += 1
                if args_match(pred["arguments"], gold.get("arguments", {})):
                    full_ok += 1

        n = len(examples)
        return {"tool_acc": tool_ok / n, "arg_acc": full_ok / n}
