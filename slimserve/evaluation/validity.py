"""Lightweight tool-call validity check (Phase 1 stand-in).

Week 1 only needs to confirm the model emits *well-formed* tool calls under load,
not full BFCL accuracy. This heuristic parses Qwen-style ``<tool_call>{...}</tool_call>``
(or a bare JSON object) and checks it names a tool with arguments.

Replaced by the real BFCLEvaluator in Week 2.
"""
from __future__ import annotations

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def is_valid_tool_call(text: str) -> bool:
    match = _TOOL_CALL_RE.search(text) or _BARE_JSON_RE.search(text)
    if not match:
        return False
    try:
        obj = json.loads(match.group(1) if match.re is _TOOL_CALL_RE else match.group(0))
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "name" in obj
