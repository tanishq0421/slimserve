# Run log — teacher_fp16 (Phase 1, Week 1)

Curated summary of the run (the full vLLM stdout is thousands of lines; only the
meaningful bits are kept here — no secrets, no dependency-resolver spam).

- **Date:** 2026-08-27
- **Hardware:** Kaggle, 2× NVIDIA Tesla T4 (16 GB each), tensor-parallel size 2
- **Model:** `Qwen/Qwen2.5-7B-Instruct`, dtype float16, `max_model_len=8192`
- **Command:** `python -m scripts.run_benchmark --config configs/teacher_fp16.yaml`

## Key log lines

```
non-default args: {'dtype': 'float16', 'tensor_parallel_size': 2,
                   'max_num_seqs': 256, 'model': 'Qwen/Qwen2.5-7B-Instruct'}
Resolved architecture: Qwen2ForCausalLM
FlashAttention 2 unavailable on T4 (compute capability 7.5)
  -> Using TRITON_ATTN attention backend
Model loading took 7.16 GiB per GPU (weights split across TP=2)
Graph capturing finished
```

## Result

```
BenchmarkResult(config_name='teacher_fp16', params_b=7.0, precision='fp16',
                tool_acc=1.0, arg_acc=1.0, tokens_per_s=728.7, ttft_ms=47.81,
                p99_latency_ms=1416.35, vram_mb=27914.0, cost_per_1m_tokens=0.1525)
```

## Notes

- `tool_acc`/`arg_acc` are the Week-1 well-formedness stand-in, not real BFCL yet.
- `vram_mb` is the sum across both T4s (nvidia-smi).
- FA2 fallback to Triton is expected on Turing — not an error.
