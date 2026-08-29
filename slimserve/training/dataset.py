"""Turn Phase-2 data records into tokenized training examples.

Renders the prompt with the student's chat template (tools included), appends the
target tool call + EOS, tokenizes, and masks the prompt tokens in the labels so
loss is computed on the completion only. Kept separate from the trainer (single
responsibility) and dependency-light so the pure logic is unit-testable.
"""
from __future__ import annotations

import json


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _prompt(record: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": record["query"]}],
        tools=record.get("tools") or None,
        add_generation_prompt=True,
        tokenize=False,
    )


def tokenize_example(record: dict, tokenizer, max_len: int) -> dict:
    """Prompt + target, tokenized, with the prompt masked out of the labels."""
    prompt = _prompt(record, tokenizer)
    completion = record["target"] + (tokenizer.eos_token or "")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    input_ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]   # -100 = ignore in the loss
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def to_dataset(records: list[dict], tokenizer, max_len: int):
    """Return an HF Dataset of tokenized, completion-masked examples."""
    from datasets import Dataset

    return Dataset.from_list(
        [tokenize_example(r, tokenizer, max_len) for r in records]
    )
