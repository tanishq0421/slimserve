"""A fixed tool-calling workload for benchmarking.

A small set of realistic (user request + available tools) scenarios, cycled up
to the requested size. Using one fixed workload across every config keeps the
benchmark table apples-to-apples.
"""
from __future__ import annotations

from slimserve.core.config import GenerationRequest


def _tool(name: str, description: str, params: dict) -> dict:
    """OpenAI-style function schema (Qwen2.5's chat template understands these)."""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


_WEATHER = _tool(
    "get_weather",
    "Get the weather forecast for a city on a date.",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
        },
        "required": ["city", "date"],
    },
)
_SEARCH = _tool(
    "search_flights",
    "Search flights between two airports on a date.",
    {
        "type": "object",
        "properties": {
            "origin": {"type": "string"},
            "destination": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["origin", "destination", "date"],
    },
)
_CALC = _tool(
    "calculate",
    "Evaluate an arithmetic expression.",
    {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)

_SCENARIOS: list[tuple[str, tuple[dict, ...]]] = [
    ("What's the weather in Delhi on 2026-08-25?", (_WEATHER,)),
    ("Find me a flight from BLR to DEL on 2026-09-01.", (_SEARCH,)),
    ("What is 18 * 47 + 3?", (_CALC,)),
    ("Will it rain in Mumbai tomorrow, 2026-08-25?", (_WEATHER, _SEARCH)),
    ("Book-search flights JFK to LHR for 2026-12-20.", (_SEARCH, _CALC)),
    ("Compute 2^10 minus 24.", (_CALC, _WEATHER)),
    ("Weather in Tokyo on 2026-10-10, please.", (_WEATHER, _SEARCH, _CALC)),
    ("Cheapest flight SFO to SEA on 2026-11-05?", (_SEARCH,)),
]


def build_tool_calling_workload(n: int, max_tokens: int = 128) -> list[GenerationRequest]:
    """Return ``n`` requests by cycling the base scenarios."""
    out: list[GenerationRequest] = []
    for i in range(n):
        prompt, tools = _SCENARIOS[i % len(_SCENARIOS)]
        out.append(
            GenerationRequest(
                prompt=prompt, tools=tools, max_tokens=max_tokens, temperature=0.0
            )
        )
    return out
