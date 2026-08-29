"""Build the gold-SFT and teacher-distillation training sets from xLAM.

Each output record is one JSONL line:
    {"query": str, "tools": [openai-format], "target": "<tool_call>...</tool_call>"}

`target` is the assistant tool-call string the student learns to emit. Records are
model-agnostic — the trainer renders the prompt with the specific student's chat
template and appends `target` as the label.
"""
from __future__ import annotations

import json
from pathlib import Path

from slimserve.core.config import GenerationRequest
from slimserve.core.interfaces import InferenceEngine
from slimserve.data.xlam import training_examples
from slimserve.evaluation.parse import parse_tool_call, xlam_tools_to_openai


def _tool_call_text(name: str, arguments: dict) -> str:
    """The Qwen-style assistant tool call the student should produce."""
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{payload}\n</tool_call>"


def _write(records: list[dict], out_path: str) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


def build_gold(num: int, out_path: str) -> int:
    """Targets = xLAM ground-truth tool calls (for the SFT baseline)."""
    records = [
        {
            "query": query,
            "tools": xlam_tools_to_openai(tools),
            "target": _tool_call_text(gold["name"], gold.get("arguments", {})),
        }
        for query, tools, gold in training_examples(num)
    ]
    return _write(records, out_path)


def build_teacher(num: int, engine: InferenceEngine, out_path: str,
                  only_correct: bool = True) -> int:
    """Targets = the teacher's own tool calls (for sequence distillation).

    With ``only_correct=True`` we keep only completions where the teacher picked
    the right tool — distilling from the teacher's good answers, not its mistakes.
    """
    examples = training_examples(num)
    requests = [
        GenerationRequest(
            prompt=query,
            tools=tuple(xlam_tools_to_openai(tools)),
            max_tokens=256,
            temperature=0.0,
        )
        for query, tools, _gold in examples
    ]
    outputs = engine.generate_batch(requests)   # batched teacher inference

    records: list[dict] = []
    for (query, tools, gold), out in zip(examples, outputs):
        pred = parse_tool_call(out.text)
        if pred is None:
            continue
        if only_correct and pred["name"] != gold.get("name"):
            continue
        records.append({
            "query": query,
            "tools": xlam_tools_to_openai(tools),
            "target": _tool_call_text(pred["name"], pred.get("arguments", {})),
        })
    return _write(records, out_path)
