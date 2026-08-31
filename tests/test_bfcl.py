"""BFCL evaluator tests — run the real vendored checker on real BFCL data with a
mock engine, so the whole pipeline (load -> request -> parse -> check -> aggregate)
is verified without a GPU."""
import json

from slimserve.core.config import GenerationOutput
from slimserve.evaluation.bfcl_evaluator import BFCLEvaluator, _user_prompt


def _gold_output(ground_truth) -> str:
    """Build a correct `<tool_call>` string from BFCL's possible-answer key."""
    blocks = []
    for call in ground_truth:
        for func, args in call.items():
            picked = {}
            for arg, acceptable in args.items():
                val = next((v for v in acceptable if v != ""), acceptable[0])
                if val == "":
                    continue                       # optional param -> omit
                picked[arg] = val
            blocks.append("<tool_call>" + json.dumps({"name": func, "arguments": picked})
                          + "</tool_call>")
    return "".join(blocks)


class _MockEngine:
    def __init__(self, by_prompt):
        self._by_prompt = by_prompt

    def generate_batch(self, requests):
        return [GenerationOutput(text=self._by_prompt.get(r.prompt, ""),
                                 prompt_tokens=1, completion_tokens=1, latency_ms=1.0)
                for r in requests]

    def generate(self, request): raise NotImplementedError
    def memory_footprint(self): raise NotImplementedError


def test_gold_outputs_pass_the_real_checker():
    ev = BFCLEvaluator(num_samples=25, categories=("simple",))
    by_prompt = {_user_prompt(c["question"]): _gold_output(gt) for c, gt in ev._load("simple")}
    metrics = ev.evaluate(_MockEngine(by_prompt))
    # Gold-derived calls should overwhelmingly satisfy BFCL's AST checker.
    assert metrics["arg_acc"] > 0.8
    assert metrics["tool_acc"] > 0.9


def test_empty_outputs_score_zero():
    ev = BFCLEvaluator(num_samples=10, categories=("simple", "multiple"))
    metrics = ev.evaluate(_MockEngine({}))
    assert metrics["arg_acc"] == 0.0
    assert metrics["tool_acc"] == 0.0


def test_parallel_needs_all_calls():
    # A single call on a parallel case must NOT count as a name match.
    ev = BFCLEvaluator(num_samples=15, categories=("parallel",))
    pairs = ev._load("parallel")
    # feed only the FIRST call of each parallel ground truth
    by_prompt = {_user_prompt(c["question"]): _gold_output([gt[0]]) for c, gt in pairs}
    metrics = ev.evaluate(_MockEngine(by_prompt))
    assert metrics["tool_acc"] < 0.5          # partial calls shouldn't match
