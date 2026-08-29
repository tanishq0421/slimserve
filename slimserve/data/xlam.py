"""Load and slice Salesforce xLAM-function-calling-60k for training.

The evaluator scores on the LAST `EVAL_HELDOUT` single-call examples, so training
draws from the FRONT of the dataset — the two never overlap, so there's no leakage
(training on your eval set would inflate the numbers).
"""
from __future__ import annotations

import json

DATASET = "Salesforce/xlam-function-calling-60k"
EVAL_HELDOUT = 200   # must match the evaluator's held-out tail


def training_examples(num: int, split: str = "train") -> list[tuple[str, list, dict]]:
    """Single-tool-call examples from the FRONT of the dataset.

    Returns a list of (query, tools, gold_answer). Stops before the held-out eval
    tail so training and evaluation never share examples.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    usable = len(ds) - EVAL_HELDOUT            # never touch the eval tail
    out: list[tuple[str, list, dict]] = []
    for i in range(usable):
        row = ds[i]
        try:
            tools = json.loads(row["tools"])
            answers = json.loads(row["answers"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if len(answers) != 1:                  # keep the target clean: one call
            continue
        out.append((row["query"], tools, answers[0]))
        if len(out) >= num:
            break
    return out
