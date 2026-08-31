"""BFCL evaluator — external, standardized function-calling accuracy (Phase 2).

Scores the engine on the **Berkeley Function-Calling Leaderboard** AST categories
(`simple`, `multiple`, `parallel`) using BFCL's own vendored test data and checker
(`_bfcl_ast.py`). This is the recognized methodology, so — unlike the self-made
xLAM exact-match metric — the numbers are externally comparable.

Same Evaluator seam, engine interface, and tool-call parser as the xLAM evaluator;
only the data and the scorer differ. Reports per-category so the parallel gap (our
students were fine-tuned on single calls only) is explicit.

  tool_acc = right function name(s), order-insensitive.
  arg_acc  = BFCL's strict AST check passes (right function AND right arguments).
"""
from __future__ import annotations

import json
from pathlib import Path

from slimserve.core.config import GenerationRequest
from slimserve.core.interfaces import Evaluator, InferenceEngine
from slimserve.core.registry import register
from slimserve.evaluation._bfcl_ast import Language, ast_checker
from slimserve.evaluation.parse import parse_tool_calls

_DATA = Path(__file__).parent / "bfcl_data"
_FILES = {
    "simple": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _user_prompt(question) -> str:
    """BFCL `question` is [[turn messages]]; take the (single-turn) user content."""
    turn = question[0]
    return next((m["content"] for m in turn if m.get("role") == "user"), turn[-1]["content"])


def _to_openai_tools(functions) -> tuple[dict, ...]:
    """BFCL function defs -> the OpenAI/function shape the chat template renders."""
    return tuple({"type": "function", "function": f} for f in functions)


@register("evaluator", "bfcl")
class BFCLEvaluator(Evaluator):
    def __init__(self, num_samples: int = 200,
                 categories=("simple", "multiple", "parallel"),
                 max_tokens: int = 512) -> None:
        self.num_samples = num_samples
        self.categories = tuple(categories)
        self.max_tokens = max_tokens

    def _load(self, category: str) -> list[tuple[dict, list]]:
        cases = _read_jsonl(_DATA / _FILES[category])[: self.num_samples]
        answers = {a["id"]: a["ground_truth"]
                   for a in _read_jsonl(_DATA / "possible_answer" / _FILES[category])}
        return [(c, answers[c["id"]]) for c in cases if c["id"] in answers]

    def evaluate(self, engine: InferenceEngine) -> dict[str, float]:
        total_name = total_valid = total_n = 0
        for category in self.categories:
            pairs = self._load(category)
            if not pairs:
                continue
            requests = [
                GenerationRequest(
                    prompt=_user_prompt(case["question"]),
                    tools=_to_openai_tools(case["function"]),
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                for case, _gt in pairs
            ]
            outputs = engine.generate_batch(requests)

            name_ok = valid_ok = 0
            for (case, gold), out in zip(pairs, outputs):
                calls = parse_tool_calls(out.text)
                model_output = [{c["name"]: c["arguments"]} for c in calls]
                pred_names = sorted(c["name"] for c in calls)
                gold_names = sorted(name for ans in gold for name in ans)
                if pred_names == gold_names:
                    name_ok += 1
                try:
                    result = ast_checker(case["function"], model_output, gold,
                                         Language.PYTHON, category, "generic")
                    valid_ok += int(bool(result.get("valid")))
                except Exception:
                    pass                                   # malformed output -> invalid

            n = len(pairs)
            total_name += name_ok
            total_valid += valid_ok
            total_n += n
            print(f"  bfcl {category:<9} tool {name_ok / n:.3f}  arg {valid_ok / n:.3f}"
                  f"  (n={n})", flush=True)

        if total_n == 0:
            return {"tool_acc": 0.0, "arg_acc": 0.0}
        print(f"  bfcl OVERALL   tool {total_name / total_n:.3f}"
              f"  arg {total_valid / total_n:.3f}", flush=True)
        return {"tool_acc": total_name / total_n, "arg_acc": total_valid / total_n}
