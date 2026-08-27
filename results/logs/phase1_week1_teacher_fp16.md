# Run log — teacher_fp16 (Phase 1, Week 1)

Curated summary of the run — the real numbers pulled from the vLLM output, with the
noise (progress bars, the huge config dump, backend warnings) left out.

- **Date:** 2026-08-27
- **Hardware:** Kaggle, 2× NVIDIA Tesla T4 (16 GB each), tensor-parallel size 2
- **Model:** `Qwen/Qwen2.5-7B-Instruct`, dtype float16, `max_model_len=8192`
- **vLLM:** v0.28.0
- **Command:** `python -m scripts.run_benchmark --config configs/teacher_fp16.yaml`

## Startup

- Weights are natively bf16; vLLM cast them to fp16 (T4 has no bf16 path).
- FlashAttention 2 isn't available on the T4 (compute capability 7.5), so it used the
  **Triton** attention backend instead. Expected, not a problem.
- Weight download ~63 s; checkpoint is 14.19 GiB across 4 shards.
- Model load: **7.16 GiB per GPU** (weights split across the two T4s), ~96 s.
- `torch.compile` warmup: ~28 s. Total engine init (profile + KV cache + warmup): ~64 s.

## Per-GPU memory breakdown (at 0.85 utilization ≈ 12.38 GiB budget)

| Piece | Size |
|---|---|
| Weights + non-torch | 7.54 GiB |
| Peak activation | 0.51 GiB |
| CUDA graphs | 1.12 GiB |
| KV cache | 4.32 GiB |

KV cache held **161,728 tokens** → up to ~19.7 concurrent requests at 8k context.
Roughly 13.5 GiB per card × 2 ≈ the ~27.9 GB `vram_mb` reported (nvidia-smi sums both).

## Result

```
BenchmarkResult(config_name='teacher_fp16', params_b=7.0, precision='fp16',
                tool_acc=1.0, arg_acc=1.0, tokens_per_s=728.7, ttft_ms=47.81,
                p99_latency_ms=1416.35, vram_mb=27914.0, cost_per_1m_tokens=0.1525)
```

## Notes

- `tool_acc`/`arg_acc` are the Week-1 well-formedness stand-in, not real BFCL yet.
- `vram_mb` is the sum across both T4s.
- The benchmark itself: 1 warmup request, one 64-request batch for throughput, then
  16 single requests for latency (p99) and 16 prefill-only requests for the TTFT proxy.
