# Run log — INT8 & INT4 quantization (Phase 1, Week 2)

Curated summary (full vLLM stdout omitted).

- **Date:** 2026-08-27
- **Hardware:** Kaggle, single NVIDIA Tesla T4 (16 GB) per run (`tensor_parallel_size=1`)
- **Models:** `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8`, `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Commands:**
  - `python -m scripts.run_benchmark --config configs/teacher_int8.yaml`
  - `python -m scripts.run_benchmark --config configs/teacher_int4_awq.yaml`

## Results

```
BenchmarkResult(config_name='teacher_int8', precision='int8', tool_acc=1.0,
                tokens_per_s=565.28, ttft_ms=42.16, p99_latency_ms=1439.43,
                vram_mb=12300.0, cost_per_1m_tokens=0.0983)
BenchmarkResult(config_name='teacher_int4_awq', precision='int4', tool_acc=1.0,
                tokens_per_s=596.77, ttft_ms=34.29, p99_latency_ms=1121.21,
                vram_mb=12328.0, cost_per_1m_tokens=0.0931)
```

## Findings

- Both quantized configs fit on a **single T4**; FP16 needed two → quantization halves
  the hardware and cuts $/1M ~1.6× (0.1525 -> 0.0931).
- INT4 (AWQ) also has the best latency (TTFT 34 ms, p99 1121 ms).
- INT8 and INT4 report near-identical `vram_mb` because vLLM fills the GPU with KV cache
  up to `gpu_memory_utilization=0.90`; lower-precision weights buy KV-cache/concurrency
  headroom, not a smaller total footprint. (TODO: measure weights-only memory to expose
  the true weight reduction.)
- `tool_acc` is still the Week-1 well-formedness stand-in; BFCL replaces it next.
