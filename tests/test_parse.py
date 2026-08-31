"""Tests for tool-call parsing — single and parallel (multi-call)."""
from slimserve.evaluation.parse import parse_tool_call, parse_tool_calls


def test_single_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    assert parse_tool_call(text) == {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert len(parse_tool_calls(text)) == 1


def test_parallel_calls():
    text = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        '<tool_call>{"name": "get_time", "arguments": {"tz": "CET"}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["get_weather", "get_time"]
    assert parse_tool_call(text)["name"] == "get_weather"   # first, back-compat


def test_nested_arguments_survive():
    text = '<tool_call>{"name": "f", "arguments": {"filter": {"min": 1, "max": 9}}}</tool_call>'
    assert parse_tool_calls(text)[0]["arguments"] == {"filter": {"min": 1, "max": 9}}


def test_no_call():
    assert parse_tool_calls("I cannot help with that.") == []
    assert parse_tool_call("nothing here") is None
