"""Turn Phase-2 data records into training text.

A record is {query, tools, target}. We render the *prompt* with the student's own
chat template (so tools are formatted exactly as the model expects), then append
the target tool call + EOS. Kept separate from the trainer (single responsibility)
and free of heavy deps so the pure logic is unit-testable.
"""
from __future__ import annotations

import json


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def render_example(record: dict, tokenizer) -> str:
    """Full training text: chat-templated prompt (with tools) + target + EOS."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": record["query"]}],
        tools=record.get("tools") or None,
        add_generation_prompt=True,
        tokenize=False,
    )
    eos = tokenizer.eos_token or ""
    return prompt + record["target"] + eos


def build_dataset(records: list[dict], tokenizer):
    """Return an HF Dataset with a single ``text`` column ready for SFT."""
    from datasets import Dataset

    texts = [render_example(r, tokenizer) for r in records]
    return Dataset.from_dict({"text": texts})
