# Phase 1 notes — quantizing a 7B tool-calling model

Working notes, not a polished report. The point is to write down what I actually
learned and, just as importantly, what these numbers do *not* prove.

## What I was trying to find out

Take `Qwen2.5-7B-Instruct`, serve it three ways — full precision (FP16), INT8, and
INT4 — and see what compression actually buys you when the job is tool/function
calling. The number I care about most is cost per million tokens, but I also tracked
throughput, latency, memory, and whether the model still emits valid tool calls.

## What the numbers said

| config | precision | GPUs | tok/s | TTFT | p99 | VRAM | $/1M |
|---|---|---|---|---|---|---|---|
| teacher_fp16 | fp16 | 2× T4 | 728.7 | 47.8 ms | 1416 ms | 27.9 GB | 0.1525 |
| teacher_int8 | int8 | 1× T4 | 565.3 | 42.2 ms | 1439 ms | 12.3 GB | 0.0983 |
| teacher_int4_awq | int4 | 1× T4 | 596.8 | 34.3 ms | 1121 ms | 12.3 GB | 0.0931 |

Full precision only runs if I split it across both T4s — 14 GB of weights doesn't fit
on one 16 GB card. Once I quantize, the whole thing fits on a single T4, and the cost
per million tokens drops from about $0.15 to $0.09 — roughly 1.6× cheaper. INT4 came
out slightly ahead of INT8 on both cost and latency, which mildly surprised me; I'd
half-expected INT8 to be quicker.

Tool-call validity stayed at 100% across all three.

The most useful thing I took away: quantization isn't just "smaller files." It changes
what hardware you need. FP16 needs two GPUs; INT4 needs one. That halving of hardware
is the real saving — more than any throughput difference.

## Things that surprised me

- **INT8 and INT4 use almost the same VRAM (~12.3 GB each).** That threw me until I
  looked closer: vLLM grabs ~90% of the GPU and fills whatever the weights don't use
  with KV cache. So lighter weights don't shrink the footprint — they leave more room
  for KV cache and more concurrent requests. Worth measuring the weights-only memory
  separately later to show the true reduction.
- **INT4 beat INT8 on latency.** Less data to move per weight, most likely.

## What this does NOT prove (being honest with myself)

This is the part I don't want to oversell.

1. **`tool_acc = 1.0` is not real accuracy.** Right now it only checks that the output
   is well-formed JSON naming a tool. It does *not* check that the model picked the
   right tool with the right arguments. So "quantization didn't hurt quality" is not
   something I've actually shown — that needs the real BFCL benchmark, which is next.
2. **The throughput comparison isn't fair.** FP16 ran on two GPUs, the quantized ones
   on one. "FP16 is faster" mostly means "two GPUs beat one," not that quantization is
   slow. The honest framing: you can't even run FP16 on one T4, so it's really
   "two GPUs of FP16" vs "one GPU of INT4."
3. **The cost figure is a reference price, not a bill.** I used $0.20 per T4-hour. Fine
   for comparing rows against each other; don't read it as a real cloud cost.
4. **The workload is tiny and synthetic** — eight hand-written prompts cycled up to 64.
   Real traffic has longer, more varied prompts, so these throughput/latency numbers
   are directional, not production numbers.
5. **One run each, no repeats.** I haven't measured how much the numbers wobble.
6. **The T4 is old (2018).** No FlashAttention 2, no native FP8. On newer hardware the
   picture — especially throughput and FP8 KV-cache — would look different.
7. **I used Qwen's official pre-quantized checkpoints.** So this shows I can *serve*
   quantized models, not that I ran the quantization pipeline myself yet.

## Bottom line

For a narrow task like tool calling, quantizing a 7B model to INT4 looks close to free
money: about 1.6× cheaper to serve, better latency, and it fits on a single mid-tier
GPU instead of two — and I couldn't find a downside in the (admittedly shallow) quality
check.

The catch is that last clause. The whole argument rests on "quality holds up," and my
current check is too weak to back that claim. So the next job is the real accuracy
benchmark (BFCL) — until that's in, the cost win is only half the story.
