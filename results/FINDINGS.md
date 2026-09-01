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

# Phase 2 notes — fine-tuning a small SLM

## What I was trying to find out

The base 1.5B already nearly matched the 7B teacher out of the box; the only real gap was
argument accuracy (0.770 vs the teacher's 0.795). So: can QLoRA fine-tuning close that gap,
and does distilling from the teacher beat plain SFT on the ground-truth labels?

Setup: QLoRA (4-bit base + LoRA adapters) on 5,000 xLAM examples, 1 epoch. Two recipes —
**gold** (train on the dataset's true tool calls) and **distill** (train on the 7B teacher's
own tool calls). Train slice is the front of xLAM; eval is the held-out tail — no overlap.

## What the numbers said

| 1.5B model | tool_acc | arg_acc | $/1M |
|---|---|---|---|
| base (off-the-shelf) | 0.985 | 0.770 | 0.038 |
| gold (SFT) | 1.000 | 0.795 | 0.038 |
| distill | 1.000 | 0.790 | 0.039 |
| *(7B teacher)* | *0.990* | *0.795* | *0.176* |

**1. Fine-tuning matched the 7B teacher with a 1.5B.** Both recipes lifted the student to
tool 1.00 / arg 0.795 — level with (or a hair past) the teacher — at ~4.5× lower cost and
lower latency. A sub-2B model doing tool-calling as well as a 7B is the whole thesis, and it
held up.

**2. Distillation did NOT beat plain SFT** (gold 0.795 vs distill 0.790 — a tie). This is the
interesting, tested-not-assumed part. Sequence distillation only helps when the teacher is a
meaningfully better source than the labels. Here the base student was *already* near the
teacher, so the teacher's answers carried no extra signal over the ground truth — and gold SFT,
being simpler, is the better default. If the student had started far behind (a genuinely weak
model), distillation would likely have earned its keep. Worth testing on the 0.5B, which has
the bigger gap.

## What this still doesn't prove

1. **`tool_acc = 1.00` reads suspiciously perfect.** It means the right tool on all 200
   held-out prompts. There's no leakage (train = front, eval = tail), and it's a narrow task,
   so it's plausible — but at n=200 treat "1.00 vs 0.99" as a tie, not a win.
2. **`arg_acc` is still strict exact-match**, so 0.795 is a floor; the true argument quality is
   a bit higher for all of them.
3. **One run, one epoch, 5k examples.** More data / a weaker student / a harder eval could all
   shift the gold-vs-distill picture.

## Then the 0.5B — does distillation help when the student starts further behind?

The 1.5B result had a caveat: the base student was *already* near the teacher, so distillation
had nothing extra to teach. The 0.5B is the cleaner test — it starts a full **8 points** behind on
arguments (0.715 vs the teacher's 0.795), so if distillation is ever going to beat plain SFT, this
is where it should show. Same recipe, same 5k examples, same two recipes (gold vs distill).

| 0.5B model | tool_acc | arg_acc | $/1M |
|---|---|---|---|
| base (off-the-shelf) | 0.975 | 0.715 | 0.017 |
| gold (SFT) | 0.990 | 0.755 | 0.016 |
| distill | 0.990 | 0.760 | 0.016 |
| *(7B teacher)* | *0.990* | *0.795* | *0.176* |

**1. Fine-tuning helped — by about 4 points.** Both recipes lifted arguments from 0.715 to ~0.755–0.76
and tool selection from 0.975 to 0.99, at **~10–11× lower cost** than the teacher (~$0.016/1M),
3,400 tok/s, ~270 ms p99. So even the smallest model gets meaningfully better at the task with a
20-minute fine-tune.

**2. Distillation still did NOT beat plain SFT** (gold 0.755 vs distill 0.760 — a 0.5-point gap, pure
noise at n=200). This is the more surprising half: I expected the bigger starting gap to finally give
the teacher's answers an edge over the ground-truth labels, and it didn't. Sequence distillation copies
the teacher's *final tokens* — and for this narrow, single-call task those tokens are essentially the
same as the gold labels the teacher was itself trained toward. Without the teacher's *probabilities*
(logit KD), there's no richer signal to transfer. So across **both** student sizes, gold SFT is the
better default: same result, simpler pipeline, no teacher-generation step.

**3. The 0.5B hit a ceiling the 1.5B didn't.** Fine-tuned, the 1.5B reached the teacher's 0.795; the
0.5B plateaued at ~0.76 and couldn't close the last ~3–4 points no matter which recipe. That's a
**capacity** limit, not a training-recipe one — the smaller model simply has less room to represent the
argument structure. Useful framing for a deployment choice: the 1.5B *matches* the 7B, the 0.5B gets
you ~95% of the way at half the size and ~10× lower cost — pick by how much that last few points is
worth.

## And finally logit KD — the richest signal, tested

Sequence KD tied gold, but with a caveat I kept flagging: it only copies the teacher's *output tokens*,
which on a single-call task are ~the gold labels. The honest test is **logit KD** — train the student to
match the teacher's *full probability distribution* per token (KL divergence, temperature 2.0, blended
50/50 with CE), using the teacher's top-50 logits precomputed offline. This is the one recipe that gives
the student something the hard labels can't: the teacher's *uncertainty* and which tools it finds similar.

| 1.5B recipe | what the student copies | tool_acc | arg_acc |
|---|---|---|---|
| gold (SFT) | the dataset's labels | 1.00 | 0.795 |
| sequence KD | the teacher's output tokens | 1.00 | 0.790 |
| **logit KD** | the teacher's full distribution | 1.00 | **0.790** |
| *(7B teacher)* | — | *0.990* | *0.795* |

**It tied too.** All three recipes — labels, teacher tokens, teacher distribution — land at ~0.79, level
with the teacher. Escalating the *richness* of the distillation signal, from hard labels all the way to
the full soft distribution, moved the number by noise. That's the clean, tested-not-assumed result:

**Why distillation's advantage didn't show up here.** "Dark knowledge" — the extra information in a soft
label — only exists when the teacher's distribution is *spread out* (a "2" that looks a bit like a "7").
For single tool-call selection the teacher is **nearly one-hot**: it's ~certain which function to call, so
`softmax(teacher)` is a spike ≈ the one-hot gold label. There's almost no distribution to transfer, so KD
collapses to SFT. Distillation earns its keep on ambiguous, high-entropy targets (open-ended generation,
noisy labels, a genuinely weak student) — not a narrow, near-deterministic task where a small student is
already competent.

## Bottom line (Phase 2)

For tool-calling, a **fine-tuned 1.5B matches the 7B teacher at a fraction of the cost**, and a
**0.5B gets ~95% of the way at ~10× lower cost** — the compression thesis, end to end and across two
sizes. And the distillation question got a thorough, escalating answer: **across all three recipes —
gold SFT, sequence KD, and logit KD — none beat plain SFT.** Not because distillation is useless, but
because this task is narrow and near-deterministic, so even the teacher's full distribution carries no
signal the labels don't. Gold SFT is the right default here; distillation would earn its place on a
higher-entropy task or a genuinely weaker student. The smallest model's remaining gap is a **capacity
ceiling**, not a recipe problem. Every one of these is a *measured* conclusion, not an assumed one.

# BFCL notes — external validation

## What I was trying to find out

Every accuracy number above is from *my own* xLAM exact-match evaluator — a strict, self-made
metric. The honest question: how do the models do on the **Berkeley Function-Calling Leaderboard
(BFCL)**, the recognized external benchmark, scored by *its* official AST checker? I ran the
`simple` / `multiple` / `parallel` categories (200 each) on the fine-tuned 1.5B gold student and the
base Qwen2.5-1.5B as an anchor, using BFCL's own vendored data + checker.

## What the numbers said

| 1.5B model | simple | multiple | parallel | **overall** |
|---|---|---|---|---|
| base (off-the-shelf) | 0.995 / 0.930 | 0.985 / 0.860 | **0.910** / 0.830 | **0.963 / 0.873** |
| gold (fine-tuned) | 0.990 / 0.920 | 0.990 / 0.880 | **0.890** / 0.830 | **0.957 / 0.877** |

*(tool / arg per category; tool = right function name(s), arg = BFCL's strict AST check passes.)*

**1. My xLAM metric was a floor — now confirmed by the standard checker.** On BFCL the fine-tuned
student scores **~0.88 arg**; my exact-match xLAM metric gave the *same model* **~0.79**. BFCL's
checker accepts valid value variations (case, acceptable ranges, optional-arg omission) that my
strict exact-match rejected. So the models are genuinely *better* than my own metric said — exactly
the caveat I flagged, now measured against the recognized methodology.

**2. Externally strong.** A fine-tuned 1.5B at **0.96 tool / 0.88 arg on BFCL** is real function-calling
competence on the standard benchmark — not a self-graded number.

**3. Base ≈ fine-tuned on the external benchmark** (0.963 vs 0.957 tool; 0.873 vs 0.877 arg). Fine-tuning
barely moved the BFCL number — consistent with the whole project's theme that the base Qwen2.5-1.5B was
*already* a strong caller. My xLAM metric showed a fine-tuning gain (0.770 → 0.795) that BFCL's more
lenient, standard checker mostly washes out: the base was already getting those args "right enough."

**4. Single-call fine-tuning mildly narrowed the model.** On **parallel** calls, the fine-tuned student
drops to **0.890** tool selection vs the base's **0.910**. We trained only on single-call xLAM, and it
cost a little parallel-calling ability — the "traded generality for single-call sharpness" effect, made
visible on a standard benchmark. It's small (not catastrophic forgetting), but it's a real, honest
trade-off worth naming.

## What this still doesn't prove

- **AST categories only.** I ran `simple`/`multiple`/`parallel` (offline, deterministic). I did not run
  `executable` (needs live APIs) or `multi-turn` (stateful) — our single-call-tuned students would fare
  worse there, and they need infra out of scope here.
- **200 per category, one run.** Directional, not a leaderboard submission.
- **Vendored checker.** I use BFCL's official `ast_checker` code and data, but stripped its model-name
  mapping (a passthrough for us) and the Java/JS type converters (Python categories only). The scoring
  logic is unchanged.

## Bottom line (BFCL)

On the standard benchmark, the fine-tuned 1.5B is a **strong function-caller (~0.96/0.88)** — and the
external check *validates* two things I'd only claimed: my own metric was a conservative floor, and the
base model was already good, so fine-tuning's real effect here is a **sharpen-single-calls, slightly-dull-
parallel** trade rather than a broad capability gain. Honest, externally comparable, and more informative
than a suspiciously clean self-made score.
