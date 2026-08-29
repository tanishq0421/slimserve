# Phase 1 notes — quantizing a 7B tool-calling model

Working notes, not a polished report. What I actually learned, and what these
numbers still don't prove.

## What I was trying to find out

Take `Qwen2.5-7B-Instruct`, serve it three ways — full precision (FP16), INT8, and
INT4 — and see what compression buys you for tool/function calling: cost, speed,
memory, and (the important one) whether the model stays accurate.

## What the numbers said

| config | precision | GPUs | tool_acc | arg_acc | tok/s | TTFT | p99 | VRAM | $/1M |
|---|---|---|---|---|---|---|---|---|---|
| teacher_fp16 | fp16 | 2× T4 | 0.990 | 0.795 | 631 | 55 ms | 4655 ms | 27.5 GB | 0.176 |
| teacher_int8 | int8 | 1× T4 | 0.985 | 0.795 | 548 | 44 ms | 1522 ms | 13.4 GB | 0.101 |
| teacher_int4_awq | int4 | 1× T4 | 0.990 | 0.780 | 577 | 37 ms | 1278 ms | 14.4 GB | 0.096 |

Two headlines:

**1. Cost and hardware.** FP16 needs two T4s — 14 GB of weights won't fit on one 16 GB
card. Quantized, it fits on one, and cost per million tokens drops from ~$0.18 to ~$0.10,
roughly 1.8× cheaper. That halving of hardware is the real saving, more than any
throughput difference.

**2. Quality held up — and this is the part I couldn't claim before.** On 200 held-out
tool-calling examples, INT4 matches FP16 on tool selection (0.99 = 0.99) and is within
about a point and a half on arguments (0.78 vs 0.795). At n=200 those gaps are inside
the noise. So for this task, dropping to 4-bit didn't cost accuracy — the cheaper,
faster model is just as good at the job.

## Things that surprised me

- **INT8 and INT4 report almost the same total VRAM (~13–14 GB).** vLLM grabs ~90% of
  the GPU and fills whatever the weights don't use with KV cache, so the *total* looks
  the same — but the split differs. Weights themselves drop a lot: FP16 ~14.2 GiB → INT8
  8.3 GiB → INT4 5.3 GiB (straight from the "Model loading took X GiB" line). The freed
  memory becomes KV cache: INT4 got 5.47 GiB (102k tokens, ~12.5× concurrency) vs INT8's
  2.51 GiB (47k tokens, ~5.7×). The real payoff of lower precision on a fixed GPU is more
  room for concurrent requests, not a smaller footprint.
- **FP16 has the *worst* tail latency despite the most throughput — but it's an artifact,
  not an FP16 cost.** FP16 p99 is ~4.7 s vs INT4's ~1.3 s. FP16 7B only ran because it was
  split across two T4s (tensor parallelism), and TP makes every layer do an all-reduce to
  combine the two GPUs' partial results — every layer, every token. Kaggle's T4s have no
  fast GPU-to-GPU link (the logs disabled P2P / custom all-reduce), so that exchange is
  slow, and it dominates the tail. The single-GPU quantized rows *are* the 7B without TP —
  ~1.3–1.5 s p99 — so the ~3× gap is the communication tax, not full precision being slow.
  On a single GPU big enough to hold FP16 (e.g. an A100) the penalty vanishes entirely; we
  couldn't measure that on free Kaggle because FP16 7B doesn't fit one 16 GB T4 — which is
  exactly why TP was needed. **Takeaway: quantization's durable wins are memory and cost;
  the latency gap here is a two-weakly-linked-GPUs artifact.**
- **Marlin kernels ran fine on the T4.** I'd assumed they needed Ampere; the logs show
  vLLM picking `MarlinLinearKernel` for both AWQ and GPTQ on the T4 and running.

## What this still doesn't prove (being honest with myself)

1. **The accuracy metric is strict and narrow.** It's my own exact-match score on
   single-tool-call examples, not the official BFCL. Exact-match under-counts arguments
   that are correct but phrased differently, so `arg_acc` (~0.78–0.795) is a *floor*, not
   the true number. It also doesn't test multi-tool, parallel calls, or "should not call
   anything" cases.
2. **Small sample, single runs.** 200 eval examples and one run per config — ±1–2 points
   is noise. I saw this directly: FP16 throughput was 728 tok/s in an earlier run and 631
   here, a ~13% swing. Treat single numbers as ballpark.
3. **The cost figure is a reference price ($0.20/T4-hr), not a real bill.**
4. **The perf workload is tiny and synthetic** — 8 prompts cycled to 64. Directional, not
   production traffic.
5. **The T4 is old (2018).** No FlashAttention 2, no native FP8. Newer hardware would
   shift throughput and unlock FP8 KV-cache.
6. **I used Qwen's official pre-quantized checkpoints.** So this shows I can *serve*
   quantized models; running the quantization myself is a later step.

## Bottom line

For tool calling, quantizing this 7B to INT4 is close to free: ~1.8× cheaper, better
latency, half the hardware — and, now measured, no accuracy loss (INT4 ties FP16 on tool
choice and is within noise on arguments). Both halves of the claim — cheaper *and* just as
good — are backed by numbers, not assumed.

The honest asterisks: it's a strict, single-call, 200-example metric, one run per config,
on old hardware, using pre-quantized checkpoints. None of that breaks the conclusion for
this task — but it's why I'd call this strong signal, not the last word.
