"""Rung-0/A correctness gates: our from-scratch forward must match HF.

Two checks:
  * logits parity — our full-sequence logits equal HF's on a fixed prompt;
  * greedy parity — our argmax decode equals HF's *true* greedy (repetition
    penalty disabled, since Qwen2.5's default generation_config applies one that
    plain argmax doesn't).

Slow — downloads Qwen2.5-0.5B and runs on CPU. Marked so the fast logic tests
still run without it:  pytest -m "not slow"
"""
import pytest

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"


@pytest.mark.slow
def test_forward_logits_match_hf():
    import torch
    from transformers import AutoTokenizer

    from slimserve.engines.mini_engine.model import MiniQwen

    mini = MiniQwen(MODEL, dtype=torch.float32, device="cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids

    ours = mini.forward(ids)
    with torch.no_grad():
        ref = mini.hf(ids).logits

    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.slow
def test_greedy_matches_hf():
    import torch
    from transformers import AutoTokenizer

    from slimserve.engines.mini_engine.model import MiniQwen

    mini = MiniQwen(MODEL, dtype=torch.float32, device="cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids

    cur = ids.clone()
    for _ in range(20):                                   # naive greedy: full forward each step
        nxt = mini.forward(cur)[0, -1].argmax().view(1, 1)
        cur = torch.cat([cur, nxt], dim=1)
    ours = cur[0, ids.shape[1]:].tolist()

    with torch.no_grad():                                 # HF *true* greedy (no rep penalty)
        ref = mini.hf.generate(ids, max_new_tokens=20, do_sample=False,
                               repetition_penalty=1.0)[0, ids.shape[1]:].tolist()
    assert ours == ref
