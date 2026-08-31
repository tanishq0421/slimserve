"""KV-cache + decode-loop tests (Phase 2, Rung A).

Fast: the ContiguousKVCache bookkeeping (tiny tensors, no model).
Slow: cached greedy decode must equal the no-cache greedy decode — and since the
no-cache path is already proven equal to HF, that transitively pins the cache.
"""
import torch

from slimserve.engines.mini_engine.kv_cache import ContiguousKVCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_cache_length_advances_once_per_step():
    c = ContiguousKVCache(n_layers=2, max_len=8, n_kv_heads=1, head_dim=3,
                          device="cpu", dtype=torch.float32)
    c.allocate(seq_id=0, num_tokens=8)
    k = torch.ones(1, 1, 3)
    fk0, _ = c.append(0, layer=0, key=k, value=k)     # first layer of a token-step
    assert fk0.shape == (1, 1, 3)
    assert c._len[0] == 0                             # length holds until last layer
    c.append(0, layer=1, key=k * 2, value=k)          # last layer advances length
    assert c._len[0] == 1
    assert c.utilization() == 1 / 8                   # 1 live token of 8 reserved


def test_cache_returns_growing_prefix():
    c = ContiguousKVCache(1, 8, 1, 2, "cpu", torch.float32)   # single layer
    c.allocate(0, 8)
    fk, _ = c.append(0, 0, torch.tensor([[[1.0, 1.0]]]), torch.zeros(1, 1, 2))
    assert fk.shape == (1, 1, 2)
    fk, _ = c.append(0, 0, torch.tensor([[[2.0, 2.0]]]), torch.zeros(1, 1, 2))
    assert fk.shape == (2, 1, 2)                       # now two tokens cached
    assert torch.allclose(fk[1], torch.tensor([[2.0, 2.0]]))


def test_paged_returns_same_kv_as_contiguous():
    # Feed the identical (layer, key, value) sequence to both caches; the full
    # prefix they return must match at every step — the paged block gather is
    # equivalent to the contiguous slice.
    from slimserve.engines.mini_engine.kv_cache import PagedKVCache

    torch.manual_seed(0)
    n_layers, n_kv, hd = 3, 2, 4
    cont = ContiguousKVCache(n_layers, 32, n_kv, hd, "cpu", torch.float32)
    paged = PagedKVCache(block_size=4, num_blocks=16, n_layers=n_layers,
                         n_kv_heads=n_kv, head_dim=hd, device="cpu", dtype=torch.float32)
    cont.allocate(0, 6)
    paged.allocate(0, 6)
    for n in (6, 1, 1, 1, 1):                     # prefill 6, then 4 decode steps
        for layer in range(n_layers):
            k = torch.randn(n, n_kv, hd)
            v = torch.randn(n, n_kv, hd)
            ck, cv = cont.append(0, layer, k, v)
            pk, pv = paged.append(0, layer, k, v)
            assert torch.allclose(ck, pk)
            assert torch.allclose(cv, pv)
    # both hold 10 live tokens; paged packs tighter, so its utilization is higher
    assert paged.utilization() >= cont.utilization()


import pytest


@pytest.mark.slow
def test_paged_decode_equals_contiguous_decode():
    from transformers import AutoTokenizer

    from slimserve.engines.mini_engine.kv_cache import PagedKVCache
    from slimserve.engines.mini_engine.model import MiniQwen

    mini = MiniQwen(MODEL, dtype=torch.float32, device="cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    n_prompt = ids.shape[1]

    def greedy(cache):
        cache.allocate(0, n_prompt)
        logits = mini.forward(ids, cache=cache, seq_id=0, start_pos=0)
        out, tok_id = [], int(logits[0, -1].argmax())
        for step in range(20):
            out.append(tok_id)
            logits = mini.forward(torch.tensor([[tok_id]]), cache=cache,
                                  seq_id=0, start_pos=n_prompt + step)
            tok_id = int(logits[0, -1].argmax())
        return out

    contiguous = greedy(ContiguousKVCache(mini.n_layers, 128, mini.n_kv,
                                          mini.head_dim, "cpu", torch.float32))
    paged = greedy(PagedKVCache(8, 32, mini.n_layers, mini.n_kv,
                                mini.head_dim, "cpu", torch.float32))
    assert contiguous == paged


@pytest.mark.slow
def test_cached_decode_equals_no_cache():
    from transformers import AutoTokenizer

    from slimserve.engines.mini_engine.model import MiniQwen

    mini = MiniQwen(MODEL, dtype=torch.float32, device="cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    n_prompt, N = ids.shape[1], 20

    # (a) no-cache greedy: full recompute each step
    cur = ids.clone()
    for _ in range(N):
        cur = torch.cat([cur, mini.forward(cur)[0, -1].argmax().view(1, 1)], dim=1)
    ref = cur[0, n_prompt:].tolist()

    # (b) cached greedy: prefill once, then decode one token at a time
    cache = ContiguousKVCache(mini.n_layers, 128, mini.n_kv, mini.head_dim,
                              "cpu", torch.float32)
    cache.allocate(0, n_prompt)
    logits = mini.forward(ids, cache=cache, seq_id=0, start_pos=0)
    got, cur_tok = [], int(logits[0, -1].argmax())
    for step in range(N):
        got.append(cur_tok)
        step_ids = torch.tensor([[cur_tok]])
        logits = mini.forward(step_ids, cache=cache, seq_id=0, start_pos=n_prompt + step)
        cur_tok = int(logits[0, -1].argmax())

    assert got == ref


@pytest.mark.slow
def test_generate_batch_matches_individual():
    # Continuous batching must give each sequence exactly what it would get alone.
    # Different-length prompts exercise the ragged running set.
    from slimserve.core.config import (EngineConfig, GenerationRequest,
                                       Precision)
    from slimserve.engines.mini_engine.engine import MiniEngine

    eng = MiniEngine(EngineConfig(name="mini", model_path=MODEL, precision=Precision.FP16,
                                  extra={"device": "cpu", "max_model_len": 256}))
    reqs = [GenerationRequest(prompt=p, temperature=0.0, max_tokens=12)
            for p in ["The capital of France is",
                      "Water is made of",
                      "The opposite of hot is"]]
    batched = eng.generate_batch(reqs)
    individual = [eng.generate(r) for r in reqs]
    for b, s in zip(batched, individual):
        assert b.text == s.text


@pytest.mark.slow
def test_mini_engine_generates():
    from slimserve.core.config import (EngineConfig, GenerationRequest,
                                       Precision)
    from slimserve.engines.mini_engine.engine import MiniEngine

    cfg = EngineConfig(name="mini", model_path=MODEL, precision=Precision.FP16,
                       extra={"device": "cpu", "max_model_len": 256})
    eng = MiniEngine(cfg)
    out = eng.generate(GenerationRequest(
        prompt="What is the capital of France?", temperature=0.0, max_tokens=10))
    assert out.completion_tokens > 0
    assert isinstance(out.text, str) and out.text.strip()
