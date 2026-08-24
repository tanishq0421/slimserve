"""vLLM-backed inference engine (Phase 1).

Adapter pattern: wraps the third-party vLLM API behind our InferenceEngine
interface so the rest of the codebase never imports vllm directly.
"""
from __future__ import annotations

from typing import Iterable

from slimserve.core.config import (
    EngineConfig,
    GenerationOutput,
    GenerationRequest,
    MemoryStats,
)
from slimserve.core.interfaces import InferenceEngine
from slimserve.core.registry import register


@register("engine", "vllm")
class VLLMEngine(InferenceEngine):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._llm = None  # lazily hold the vllm.LLM handle

    def _ensure_loaded(self) -> None:
        if self._llm is None:
            # from vllm import LLM
            # self._llm = LLM(model=self.config.model_path,
            #                 quantization=..., kv_cache_dtype=...)
            raise NotImplementedError("Phase 1, Week 1: load vLLM here.")

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        return self.generate_batch([request])[0]

    def generate_batch(
        self, requests: Iterable[GenerationRequest]
    ) -> list[GenerationOutput]:
        self._ensure_loaded()
        raise NotImplementedError("Phase 1, Week 1: map requests -> vLLM.generate.")

    def memory_footprint(self) -> MemoryStats:
        raise NotImplementedError("Phase 1, Week 1: read VRAM stats.")
