"""Typed configuration and data-transfer objects.

Everything that crosses a module boundary is a frozen dataclass, not a loose
dict. This keeps interfaces explicit (you can see exactly what each stage needs)
and makes configs hashable/serializable for the results store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Precision(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"


# --- Inference DTOs ---------------------------------------------------------

@dataclass(frozen=True)
class GenerationRequest:
    """One request into any InferenceEngine."""
    prompt: str
    tools: tuple[dict, ...] = ()          # tool schemas available to the model
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass(frozen=True)
class GenerationOutput:
    """One result out of any InferenceEngine."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float | None = None          # time to first token (streaming)


@dataclass(frozen=True)
class MemoryStats:
    weights_mb: float
    kv_cache_mb: float
    peak_mb: float


# --- Stage configs ----------------------------------------------------------

@dataclass(frozen=True)
class EngineConfig:
    name: str                              # registry key, e.g. "vllm"
    model_path: str
    precision: Precision = Precision.FP16
    kv_cache_quant: bool = False
    max_num_seqs: int = 256                # continuous-batching width
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class QuantConfig:
    name: str                              # "awq" | "gptq" | "bnb"
    precision: Precision = Precision.INT4
    out_path: str = ""
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    dataset: str
    output_dir: str
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 1
    lr: float = 2e-4
    load_in_4bit: bool = True              # QLoRA
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DistillConfig:
    strategy: str                          # "sequence_kd" | "logit_kd"
    teacher_model: str
    temperature: float = 2.0               # softmax temperature for logit KD
    alpha: float = 0.5                     # weight between KD loss and task loss
    extra: dict = field(default_factory=dict)


def engine_config_from_dict(cfg: dict) -> EngineConfig:
    """Build an EngineConfig from a parsed YAML dict (shared by the CLIs)."""
    return EngineConfig(
        name=cfg["engine"],
        model_path=cfg["model_path"],
        precision=Precision(cfg.get("precision", "fp16")),
        kv_cache_quant=cfg.get("kv_cache_quant", False),
        max_num_seqs=cfg.get("max_num_seqs", 256),
        extra={
            "tensor_parallel_size": cfg.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.90),
            "max_model_len": cfg.get("max_model_len"),
            "enforce_eager": cfg.get("enforce_eager", False),
            "quantization": cfg.get("quantization"),
        },
    )


def train_config_from_dict(cfg: dict) -> TrainConfig:
    """Build a TrainConfig from a parsed YAML dict (used by the training CLI)."""
    knobs = ("max_seq_len", "batch_size", "grad_accum", "seed",
             "temperature", "alpha", "teacher_model")   # last three: logit KD
    return TrainConfig(
        base_model=cfg["base_model"],
        dataset=cfg["dataset"],
        output_dir=cfg["output_dir"],
        lora_r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        epochs=cfg.get("epochs", 1),
        lr=cfg.get("lr", 2e-4),
        load_in_4bit=cfg.get("load_in_4bit", True),
        extra={k: cfg[k] for k in knobs if k in cfg},
    )
