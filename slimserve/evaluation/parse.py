"""Parsing + scoring helpers for tool-calling accuracy.

Pure functions, no GPU/network — kept separate so they're easy to unit-test.
"""
from __future__ import annotations

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_one(raw: str) -> dict | None:
    """One JSON blob -> ``{"name", "arguments"}`` (or None if malformed)."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):                 # some models emit arguments as a string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}


def parse_tool_calls(text: str) -> list[dict]:
    """Pull ALL tool calls out of model output — needed for parallel calls.

    Finds every ``<tool_call>{...}</tool_call>`` block; if none are present, falls
    back to a single bare JSON object. Returns a list of ``{"name", "arguments"}``
    (empty if nothing parses).
    """
    raws = _TOOL_CALL_RE.findall(text)
    if not raws:
        m = _BARE_JSON_RE.search(text)
        raws = [m.group(0)] if m else []
    return [c for c in (_parse_one(r) for r in raws) if c]


def parse_tool_call(text: str) -> dict | None:
    """The first tool call, for single-call evaluators. See ``parse_tool_calls``."""
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def xlam_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert xLAM tool specs into the OpenAI/JSON-schema function format that
    Qwen's chat template expects.

    xLAM shape:  {"name", "description", "parameters": {p: {"type","description","default"?}}}
    OpenAI shape: {"type":"function","function":{"name","description","parameters": <json schema>}}
    """
    out: list[dict] = []
    for t in tools:
        params = t.get("parameters", {}) or {}
        props: dict = {}
        required: list[str] = []
        for pname, spec in params.items():
            spec = spec or {}
            prop = {"type": str(spec.get("type", "string"))}
            if spec.get("description"):
                prop["description"] = spec["description"]
            props[pname] = prop
            if "default" not in spec:          # no default => treat as required
                required.append(pname)
        schema: dict = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": schema,
            },
        })
    return out


def _norm(v) -> str:
    return str(v).strip().lower()


def args_match(pred: dict, gold: dict) -> bool:
    """Strict, normalized exact match: same keys, same values (case-insensitive)."""
    if set(pred.keys()) != set(gold.keys()):
        return False
    return all(_norm(pred[k]) == _norm(gold.get(k)) for k in gold)
