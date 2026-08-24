"""The seams of the system.

Every axis that varies in this project — how a model is served, how it is
quantized, how the KV cache is laid out, how the student is trained, how it is
evaluated — is expressed as a small abstract interface here. Concrete
implementations live in their own packages and are wired in by the registry.

Design principles this file exists to enforce:
  * DIP  — high-level code (BenchmarkRunner, Pipeline) depends on these ABCs,
           never on vLLM/AWQ/etc. directly.
  * OCP  — a new backend is a new subclass, not an edit to existing code.
  * ISP  — each interface is deliberately tiny; no fat base classes.
  * LSP  — any implementation is substitutable wherever the ABC is accepted.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from slimserve.core.config import (
    DistillConfig,
    GenerationOutput,
    GenerationRequest,
    MemoryStats,
    QuantConfig,
    TrainConfig,
)


class InferenceEngine(ABC):
    """A way of running a model for generation.

    Implementations: VLLMEngine, HFEngine, MiniEngine (from scratch).
    """

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        ...

    @abstractmethod
    def generate_batch(
        self, requests: Iterable[GenerationRequest]
    ) -> list[GenerationOutput]:
        ...

    @abstractmethod
    def memory_footprint(self) -> MemoryStats:
        ...


class Quantizer(ABC):
    """Turns a full-precision checkpoint into a lower-precision one.

    Implementations: AWQQuantizer, GPTQQuantizer, BnBQuantizer.
    """

    @abstractmethod
    def quantize(self, model_path: str, config: QuantConfig) -> str:
        """Return the path to the quantized checkpoint."""
        ...


class KVCache(ABC):
    """Phase-2 teaching abstraction: how key/value tensors are stored.

    Implementations: ContiguousKVCache (naive), PagedKVCache (blocked).
    """

    @abstractmethod
    def allocate(self, seq_id: int, num_tokens: int) -> None:
        ...

    @abstractmethod
    def append(self, seq_id: int, key, value) -> None:
        ...

    @abstractmethod
    def free(self, seq_id: int) -> None:
        ...

    @abstractmethod
    def utilization(self) -> float:
        """Fraction of allocated memory actually holding live KV data."""
        ...


class Scheduler(ABC):
    """Decides which requests run in the current decode step (Phase 2)."""

    @abstractmethod
    def admit(self, request: GenerationRequest) -> int:
        """Register a request; return its seq_id."""
        ...

    @abstractmethod
    def next_batch(self) -> list[int]:
        """Return seq_ids to advance this step (continuous batching)."""
        ...


class Trainer(ABC):
    """Produces a fine-tuned student checkpoint.

    Implementations: QLoRATrainer, and DistillationTrainer (which composes a
    DistillStrategy). Base training loop is a Template Method (see training/base).
    """

    @abstractmethod
    def train(self, config: TrainConfig) -> str:
        """Return the path to the trained checkpoint."""
        ...


class DistillStrategy(ABC):
    """How the student learns from the teacher.

    Implementations: SequenceLevelKD (train on teacher completions),
    LogitKD (KL-divergence to teacher soft labels).
    """

    @abstractmethod
    def compute_loss(self, student_batch, teacher_batch, config: DistillConfig):
        ...


class Evaluator(ABC):
    """Scores an engine on the tool-calling task.

    Implementations: BFCLEvaluator, ToolBenchEvaluator.
    """

    @abstractmethod
    def evaluate(self, engine: InferenceEngine) -> dict[str, float]:
        """Return a metrics dict, e.g. {'tool_acc': 0.91, 'arg_acc': 0.87}."""
        ...
