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


# --- Logit-KD helpers -------------------------------------------------------
# The teacher's soft labels are precomputed offline (build_logits) and stored as
# top-k logits over the *completion* tokens only. These pure functions are the
# extraction (build time) and the ragged-padding (collate time) — both GPU-free
# and unit-tested, so the on-disk alignment can't silently drift from the loss.

def extract_teacher_topk(logits, labels, k: int):
    """Top-k teacher logits at the supervised (completion) positions.

    ``logits`` [L, V], ``labels`` [L] (list or tensor, -100 on prompt). Uses the
    same next-token shift as the loss: position i predicts token i+1, so a row is
    supervised when ``labels[i+1] != -100``. Returns, in left-to-right order:
    ``vals`` [n, k] fp16 and ``ids`` [n, k] int64 — aligned token-for-token with
    how the loss re-derives the supervised mask from ``labels``.
    """
    import torch

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    shift_logits = logits[:-1]                 # [L-1, V]
    shift_labels = labels[1:]                  # [L-1]
    supervised = shift_labels != -100
    vals, ids = shift_logits[supervised].topk(k, dim=-1)   # [n, k]
    return vals.to(torch.float16), ids.to(torch.long)


def pad_teacher_topk(ids_per_example, vals_per_example):
    """Pad ragged per-example top-k tensors to a dense batch.

    Each example carries ``n`` supervised rows of width ``k`` (``n`` varies with
    completion length). Returns ``topk_ids`` [B, M, k] long, ``topk_vals`` [B, M, k]
    float, and ``kd_mask`` [B, M] bool, where M is the longest completion in the
    batch. Pad rows are zeros; the loss never reads them (it slices the first
    ``n = supervised.sum()`` rows per example), but kd_mask makes that explicit.
    """
    import torch

    B = len(ids_per_example)
    M = max((len(x) for x in ids_per_example), default=0)
    # k = width of a top-k row. Read it from the first NON-EMPTY example, since an
    # example can have zero supervised rows (its completion was truncated away).
    k = next((len(ex[0]) for ex in ids_per_example if ex), 0)
    topk_ids = torch.zeros(B, M, k, dtype=torch.long)
    topk_vals = torch.zeros(B, M, k, dtype=torch.float)
    kd_mask = torch.zeros(B, M, dtype=torch.bool)
    for b, (ids, vals) in enumerate(zip(ids_per_example, vals_per_example)):
        n = len(ids)
        if n == 0:
            continue
        topk_ids[b, :n] = torch.as_tensor(ids, dtype=torch.long)
        topk_vals[b, :n] = torch.as_tensor(vals, dtype=torch.float)
        kd_mask[b, :n] = True
    return topk_ids, topk_vals, kd_mask


def load_logit_dataset(path: str):
    """Load the Arrow dataset written by ``build_data --mode logits``.

    Drops examples whose completion was truncated to nothing (no supervised
    tokens -> empty top-k): they carry no CE or KD signal and would only add
    empty batches.
    """
    from datasets import load_from_disk

    ds = load_from_disk(path)
    return ds.filter(lambda ex: len(ex["kd_topk_ids"]) > 0)


class KDDataCollator:
    """Pad token fields like DataCollatorForSeq2Seq, plus the ragged teacher top-k.

    Splits the two concerns: the standard collator handles input_ids/attention_mask/
    labels padding; ``pad_teacher_topk`` (pure, tested) handles the KD tensors.
    """

    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        from transformers import DataCollatorForSeq2Seq

        self._base = DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=label_pad_token_id)

    def __call__(self, features):
        ids = [f.pop("kd_topk_ids") for f in features]
        vals = [f.pop("kd_topk_vals") for f in features]
        batch = self._base(features)
        topk_ids, topk_vals, kd_mask = pad_teacher_topk(ids, vals)
        batch["kd_topk_ids"] = topk_ids
        batch["kd_topk_vals"] = topk_vals
        batch["kd_mask"] = kd_mask
        return batch
