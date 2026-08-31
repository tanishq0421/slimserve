"""vLLM-backed inference engine (Phase 1).

Adapter pattern: wraps the third-party vLLM API behind our InferenceEngine
interface so the rest of the codebase never imports vllm directly.

Tool-calling prompts are formatted with the model's own chat template (including
the tool schemas) via the HF tokenizer, which is the most version-robust way to
get correct tool-calling behavior out of Qwen2.5.

NOTE: the vLLM/transformers imports are intentionally inside methods so the
package imports fine on a machine without a GPU (e.g. your laptop). This file's
GPU path is exercised on Kaggle, not in local unit tests.
"""
from __future__ import annotations

import subprocess
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


def _gpu_used_mb() -> float:
    """Sum used VRAM across all visible GPUs via nvidia-smi (Kaggle: dedicated box)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(sum(int(x) for x in out.split()))
    except Exception:
        return 0.0


@register("engine", "vllm")
class VLLMEngine(InferenceEngine):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._llm = None
        self._tokenizer = None

    # --- setup ---
    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM

        cfg = self.config
        dtype = "float16" if cfg.precision is Precision.FP16 else "auto"
        # Quantized configs point model_path at an already-quantized checkpoint and
        # name the method explicitly (e.g. "awq", "gptq"). On T4 (Turing) use the
        # plain awq/gptq kernels — the *_marlin kernels need Ampere+. FP16 baseline
        # leaves this None.
        quantization = cfg.extra.get("quantization")
        if quantization is None:
            quantization = {Precision.INT4: "awq", Precision.INT8: "gptq"}.get(cfg.precision)
        kv_cache_dtype = "fp8" if cfg.kv_cache_quant else "auto"
        tp = int(cfg.extra.get("tensor_parallel_size", 1))
        # Memory knobs (matter on 16GB T4s): leave headroom for transient warmup
        # allocations, and cap context so the KV cache reservation stays modest.
        gpu_mem_util = float(cfg.extra.get("gpu_memory_utilization", 0.90))
        max_model_len = cfg.extra.get("max_model_len")           # None -> model default
        enforce_eager = bool(cfg.extra.get("enforce_eager", False))

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        llm_kwargs = dict(
            model=cfg.model_path,
            tensor_parallel_size=tp,          # TP=2 spans both Kaggle T4s for FP16 7B
            dtype=dtype,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            max_num_seqs=cfg.max_num_seqs,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        # bitsandbytes in-flight quantization of an un-quantized checkpoint needs
        # load_format="bitsandbytes" (quantizes NF4 at load — no pre-quant step).
        load_format = cfg.extra.get("load_format")
        if load_format:
            llm_kwargs["load_format"] = load_format
        self._llm = LLM(**llm_kwargs)

    def _format(self, request: GenerationRequest) -> str:
        messages = [{"role": "user", "content": request.prompt}]
        return self._tokenizer.apply_chat_template(
            messages,
            tools=list(request.tools) or None,
            add_generation_prompt=True,
            tokenize=False,
        )

    def _sampling(self, request: GenerationRequest):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=request.temperature, max_tokens=request.max_tokens
        )

    @staticmethod
    def _to_output(vout, latency_ms: float) -> GenerationOutput:
        completion = vout.outputs[0]
        return GenerationOutput(
            text=completion.text,
            prompt_tokens=len(vout.prompt_token_ids),
            completion_tokens=len(completion.token_ids),
            latency_ms=latency_ms,
        )

    # --- inference ---
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self._ensure_loaded()
        t0 = time.perf_counter()
        outs = self._llm.generate([self._format(request)], self._sampling(request))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._to_output(outs[0], latency_ms)

    def generate_batch(
        self, requests: Iterable[GenerationRequest]
    ) -> list[GenerationOutput]:
        self._ensure_loaded()
        reqs = list(requests)
        prompts = [self._format(r) for r in reqs]
        params = [self._sampling(r) for r in reqs]
        t0 = time.perf_counter()
        outs = self._llm.generate(prompts, params)   # vLLM batches internally
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Shared batch latency is informational; the runner derives throughput
        # from the aggregate elapsed time, not per-request latency.
        return [self._to_output(o, elapsed_ms) for o in outs]

    def memory_footprint(self) -> MemoryStats:
        # nvidia-smi captures worker-process VRAM too (matters under TP>1).
        used = _gpu_used_mb()
        return MemoryStats(weights_mb=0.0, kv_cache_mb=0.0, peak_mb=used)
