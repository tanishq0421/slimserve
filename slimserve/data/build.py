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


def build_logits(num: int, teacher_model: str, out_dir: str,
                 top_k: int = 50, max_len: int = 2048) -> int:
    """Precompute the teacher's top-k logits for logit KD (offline, run once).

    Targets are the gold tool calls; for each we run the teacher forward and store
    its top-k logits over the *completion* tokens only. The teacher is loaded in
    4-bit so the 7B fits one T4, and nothing teacher-side is held at train time.
    Output is an Arrow dataset (tokenized here, so train-time tokenization can't
    drift) with columns: input_ids, attention_mask, labels, kd_topk_ids, kd_topk_vals.
    """
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    from slimserve.training.dataset import extract_teacher_topk, tokenize_example

    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model, quantization_config=quant,
        torch_dtype=torch.float16, device_map={"": 0}).eval()

    # One example at a time (batch=1) — no padding, so the top-k rows line up with
    # the labels exactly. Could be batched for speed; kept simple and correct.
    rows: list[dict] = []
    for query, tools, gold in training_examples(num):
        record = {
            "query": query,
            "tools": xlam_tools_to_openai(tools),
            "target": _tool_call_text(gold["name"], gold.get("arguments", {})),
        }
        ex = tokenize_example(record, tokenizer, max_len)
        ids = torch.tensor([ex["input_ids"]], device=model.device)
        with torch.no_grad():
            logits = model(ids).logits[0]            # [L, V]
        vals, top_ids = extract_teacher_topk(logits.float().cpu(), ex["labels"], top_k)
        rows.append({**ex,
                     "kd_topk_ids": top_ids.tolist(),
                     "kd_topk_vals": vals.float().tolist()})

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(out_dir)
    return len(rows)
