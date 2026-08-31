"""MiniEngine — the from-scratch inference engine (Phase 3).

Implements the same InferenceEngine interface as the vLLM adapter, so it drops
into the exact benchmark harness and is selected by ``engine: mini`` in a YAML.
It owns the whole serving loop: format the tool-calling prompt, prefill into a
KV cache, then greedily (or with temperature) decode token by token, reusing the
cache instead of recomputing the prompt each step.

This first cut serves one sequence at a time (``generate_batch`` just loops).
Continuous batching — many sequences sharing decode steps — is the next rung.
"""
from __future__ import annotations

import time
from typing import Iterable

from slimserve.core.config import (
    EngineConfig,
    GenerationOutput,
    GenerationRequest,
    MemoryStats,
    Precision,
)
from slimserve.core.interfaces import InferenceEngine
from slimserve.core.registry import register
from slimserve.engines.mini_engine.kv_cache import ContiguousKVCache, PagedKVCache
from slimserve.engines.mini_engine.model import MiniQwen


@register("engine", "mini")
class MiniEngine(InferenceEngine):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    # --- setup --------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer

        cfg = self.config
        self._device = cfg.extra.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = (torch.float16 if (self._device.startswith("cuda")
                                         and cfg.precision is Precision.FP16)
                       else torch.float32)
        self._max_len = int(cfg.extra.get("max_model_len") or 2048)
        self._kv_kind = cfg.extra.get("kv_cache", "contiguous")
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self._model = MiniQwen(cfg.model_path, dtype=self._dtype, device=self._device)

    def _make_cache(self):
        """Contiguous (naive) or paged KV cache, selected by ``kv_cache`` in config."""
        import math

        m = self._model
        if self._kv_kind == "paged":
            block_size = int(self.config.extra.get("block_size", 16))
            num_blocks = int(self.config.extra.get(
                "num_blocks", math.ceil(self._max_len / block_size) + 1))
            return PagedKVCache(block_size, num_blocks, m.n_layers, m.n_kv,
                                m.head_dim, self._device, self._dtype)
        return ContiguousKVCache(m.n_layers, self._max_len, m.n_kv,
                                 m.head_dim, self._device, self._dtype)

    def _format(self, request: GenerationRequest) -> str:
        messages = [{"role": "user", "content": request.prompt}]
        return self._tokenizer.apply_chat_template(
            messages, tools=list(request.tools) or None,
            add_generation_prompt=True, tokenize=False)

    def _sample(self, logits, temperature: float) -> int:
        import torch

        if temperature <= 0.0:
            return int(logits.argmax())
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))

    def _prep_ids(self, request: GenerationRequest):
        """Tokenize the tool-formatted prompt, truncated so prompt + generation fit
        the cache (the contiguous cache is sized to max_model_len; overflowing it
        would error). Long prompts lose their tail rather than crash the run."""
        prompt = self._format(request)
        ids = self._tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(self._device)
        budget = max(1, self._max_len - request.max_tokens)
        return ids[:, :budget]

    # --- inference ----------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self._ensure_loaded()
        import torch

        t0 = time.perf_counter()
        ids = self._prep_ids(request)
        n_prompt = ids.shape[1]

        model = self._model
        cache = self._make_cache()
        cache.allocate(seq_id=0, num_tokens=n_prompt)

        # prefill: run the whole prompt, fill the cache, sample the first token
        logits = model.forward(ids, cache=cache, seq_id=0, start_pos=0)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        tok = self._sample(logits[0, -1], request.temperature)

        eos = self._tokenizer.eos_token_id
        generated: list[int] = []
        for step in range(request.max_tokens):          # decode one token at a time
            generated.append(tok)
            if tok == eos:
                break
            step_ids = torch.tensor([[tok]], device=self._device)
            logits = model.forward(step_ids, cache=cache, seq_id=0,
                                   start_pos=n_prompt + step)
            tok = self._sample(logits[0, -1], request.temperature)

        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GenerationOutput(text=text, prompt_tokens=n_prompt,
                                completion_tokens=len(generated),
                                latency_ms=latency_ms, ttft_ms=ttft_ms)

    def generate_batch(
        self, requests: Iterable[GenerationRequest]
    ) -> list[GenerationOutput]:
        """Continuous batching: many sequences share one cache and advance together.

        Each request is admitted to the scheduler and prefilled on first run; then
        every decode step advances the whole running set at once. Sequences retire
        independently on EOS/max_tokens (freeing their cache blocks), and waiting
        requests take the freed slots — no head-of-line blocking.
        """
        self._ensure_loaded()
        import time

        import torch

        from slimserve.engines.mini_engine.scheduler import ContinuousBatchScheduler

        reqs = list(requests)
        if not reqs:
            return []
        t0 = time.perf_counter()
        model, eos = self._model, self._tokenizer.eos_token_id
        cache = self._make_cache()                       # shared across sequences
        scheduler = ContinuousBatchScheduler(self.config.max_num_seqs)

        state: dict[int, dict] = {}
        for r in reqs:
            sid = scheduler.admit(r)
            ids = self._prep_ids(r)
            state[sid] = {"ids": ids, "n_prompt": ids.shape[1], "req": r,
                          "gen": [], "pos": 0, "next": None}
        order = list(state.keys())                        # return in request order

        while scheduler.has_work():
            running = scheduler.next_batch()
            for sid in running:                           # prefill newcomers
                st = state[sid]
                if st["next"] is None:
                    cache.allocate(sid, st["n_prompt"])
                    logits = model.forward(st["ids"], cache=cache, seq_id=sid, start_pos=0)
                    st["pos"] = st["n_prompt"]
                    st["next"] = self._sample(logits[0, -1], st["req"].temperature)

            tokens = torch.tensor([[state[s]["next"]] for s in running], device=self._device)
            positions = [state[s]["pos"] for s in running]
            logits = model.decode_batch(tokens, cache, running, positions)   # [B,1,V]
            for i, sid in enumerate(running):
                st = state[sid]
                st["gen"].append(st["next"])
                st["pos"] += 1
                if st["next"] == eos or len(st["gen"]) >= st["req"].max_tokens:
                    cache.free(sid)
                    scheduler.retire(sid)
                else:
                    st["next"] = self._sample(logits[i, -1], st["req"].temperature)

        elapsed = (time.perf_counter() - t0) * 1000.0
        outs = []
        for sid in order:
            st = state[sid]
            outs.append(GenerationOutput(
                text=self._tokenizer.decode(st["gen"], skip_special_tokens=True),
                prompt_tokens=st["n_prompt"], completion_tokens=len(st["gen"]),
                latency_ms=elapsed))
        return outs

    def memory_footprint(self) -> MemoryStats:
        self._ensure_loaded()
        weights_mb = sum(p.numel() * p.element_size()
                         for p in self._model.hf.parameters()) / 1e6
        return MemoryStats(weights_mb=weights_mb, kv_cache_mb=0.0, peak_mb=weights_mb)
