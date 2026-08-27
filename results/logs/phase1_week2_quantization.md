# Run log — INT8 & INT4 quantization (Phase 1, Week 2)

Real numbers pulled from the two vLLM runs, cleaned of progress bars, the config
dump, and backend noise.

- **Date:** 2026-08-27
- **Hardware:** Kaggle, single NVIDIA Tesla T4 (16 GB) per run (`tensor_parallel_size=1`)
- **vLLM:** v0.28.0, `max_model_len=8192`, `gpu_memory_utilization=0.9`
- Both runs used the **Triton** attention backend (FA2 needs Ampere+; T4 is Turing).
- Both quantized checkpoints ran through vLLM's **Marlin** kernel on the T4
  (`AutoGPTQ`/`AutoAWQ` → `MarlinLinearKernel`) — so Marlin *does* run on Turing here,
  contrary to what I'd assumed.

## INT8 — `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8`

- Checkpoint 8.25 GiB; weights download ~30 s; model load **8.29 GiB**, ~40 s.
- `torch.compile` ~29 s; engine init ~70 s total.
- KV cache: **2.51 GiB → 46,960 tokens**, max concurrency **5.73×** at 8k context.

| Piece | Size |
|---|---|
| Weights + non-torch | 9.54 GiB |
| Peak activation | 1.06 GiB |
| CUDA graphs | 0.51 GiB |
| KV cache | 2.51 GiB |

```
BenchmarkResult(config_name='teacher_int8', precision='int8', tool_acc=1.0,
                tokens_per_s=565.28, ttft_ms=42.16, p99_latency_ms=1439.43,
                vram_mb=12300.0, cost_per_1m_tokens=0.0983)
```

## INT4 — `Qwen/Qwen2.5-7B-Instruct-AWQ`

- Checkpoint 5.19 GiB; weights download ~17 s; model load **5.29 GiB**, ~25 s.
- `torch.compile` ~27 s; engine init ~65 s total.
- KV cache: **5.47 GiB → 102,320 tokens**, max concurrency **12.49×** at 8k context.

| Piece | Size |
|---|---|
| Weights + non-torch | 6.58 GiB |
| Peak activation | 1.06 GiB |
| CUDA graphs | 0.52 GiB |
| KV cache | 5.47 GiB |

```
BenchmarkResult(config_name='teacher_int4_awq', precision='int4', tool_acc=1.0,
                tokens_per_s=596.77, ttft_ms=34.29, p99_latency_ms=1121.21,
                vram_mb=12328.0, cost_per_1m_tokens=0.0931)
```

## What the two runs together show

- **Weights-only memory drops a lot:** FP16 ~14.2 GiB → INT8 8.29 GiB → INT4 5.29 GiB
  (roughly 2.7× smaller than FP16 at INT4). This is the real compression, and it's
  visible in the "Model loading took X GiB" line — no separate measurement needed.
- **The freed memory becomes KV cache, not a smaller footprint.** Total `vram_mb` is
  ~12.3 GB for both because vLLM fills to `gpu_memory_utilization=0.9`. But the split
  differs sharply: INT4 gets 5.47 GiB of KV cache (12.5× concurrency) vs INT8's
  2.51 GiB (5.7×). So INT4's real edge on a fixed GPU is **more room for concurrent
  requests**, on top of being cheaper and lower-latency.
- `tool_acc`/`arg_acc` are still the Week-1 well-formedness stand-in, not BFCL.
- `vram_mb` is the single-GPU nvidia-smi total.
