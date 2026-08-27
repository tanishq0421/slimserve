"""Parsing + scoring helpers for tool-calling accuracy.

Pure functions, no GPU/network — kept separate so they're easy to unit-test.
"""
from __future__ import annotations

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_tool_call(text: str) -> dict | None:
    """Pull the first tool call out of model output.

    Handles Qwen's ``<tool_call>{...}</tool_call>`` wrapper and a bare JSON
    object fallback. Returns ``{"name": str, "arguments": dict}`` or None.
    """
    m = _TOOL_CALL_RE.search(text)
    raw = m.group(1) if m else (_BARE_JSON_RE.search(text) or [None])[0]
    if isinstance(raw, re.Match):
        raw = raw.group(0)
    if not raw:
        return None
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
