# SlimServe

**Compress a large tool-calling LLM into a fast, cheap SLM — and prove the cost delta at every step.**

Big models are expensive to serve. For a narrow, high-value task —
**agentic tool / function calling** — a well-compressed small model can match a
large one at a fraction of the cost. SlimServe takes a 7B teacher, shrinks it to
a ~1.5B student, and measures **$/1M tokens, throughput, latency, VRAM, and
tool-calling accuracy** across every configuration.

> **Headline target:** distilled INT4 SLM at **~6–8× lower cost**, **≥~90–95% of
> teacher tool-calling accuracy**.

## Results so far

![Cost, memory, and accuracy across precisions](results/charts/phase1_cost_vs_quality.png)

Live table in [`results/benchmarks.csv`](results/benchmarks.csv); per-run notes in
[`results/logs/`](results/logs/). Regenerate the chart with `python scripts/plot_results.py`.

| config | params | precision | GPUs | tool_acc | arg_acc | tok/s | TTFT (ms) | p99 (ms) | VRAM (MB) | $/1M tok |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher_fp16 | 7B | fp16 | 2× T4 | 0.990 | 0.795 | 631.1 | 55.3 | 4655 | 27462 | 0.1761 |
| teacher_int8 | 7B | int8 (GPTQ) | 1× T4 | 0.985 | 0.795 | 548.3 | 43.5 | 1522 | 13408 | 0.1013 |
| teacher_int4_awq | 7B | int4 (AWQ) | 1× T4 | 0.990 | 0.780 | 576.6 | 36.6 | 1278 | 14422 | 0.0964 |
| student_1p5b_base | 1.5B | fp16 | 1× T4 | 0.985 | 0.770 | 1470.3 | 26.6 | 670 | 13808 | 0.0378 |
| student_0p5b_base | 0.5B | fp16 | 1× T4 | 0.975 | 0.715 | 3373.0 | 23.6 | 270 | 13836 | **0.0165** |

<sub>Accuracy = tool-calling on 200 held-out xLAM examples: `tool_acc` = right tool,
`arg_acc` = right tool **and** right arguments (strict exact-match, so a floor). ±1–2 pts is
noise at n=200. Students are **off-the-shelf** (no distillation yet). Measured on Kaggle T4s;
`$/1M` uses $0.20/hr-per-T4 (FP16's 2× T4 = $0.40/hr).</sub>

**Takeaways:**

- **Quantization is essentially free** for this task: INT4 matches FP16 on tool selection
  (0.99 = 0.99), within noise on arguments, at **~1.8× lower cost** and better latency, on half
  the hardware. (Weights shrink ~14 → 8.3 → 5.3 GiB FP16→INT8→INT4; the freed memory becomes
  KV-cache headroom. FP16's p99 is ~3.6× *worse* — tensor-parallelism across two link-less T4s
  pays a per-token communication cost single-GPU INT4 avoids.)
- **You may not need the big model at all.** An **off-the-shelf 1.5B** matches the 7B teacher on
  tool selection (0.985 vs 0.990) at **~4.7× lower cost, 2.3× throughput, 7× better p99** — and
  even a **0.5B** holds 0.975 tool accuracy at **~10× lower cost**. The only real gap is argument
  accuracy on the smallest model (0.715). This is the "SLM-first agents" result on our own numbers.

Full write-up, including what these numbers *don't* prove, in
[`results/FINDINGS.md`](results/FINDINGS.md).

![Phase 1 Week 1 benchmark on Kaggle](docs/images/phase1_week1_benchmark.png)

## The story in three phases

1. **Serve & benchmark** existing tools — vLLM + INT8/INT4/FP8 quantization + KV-cache
   compression. Establish the cost baseline.
2. **Rebuild the internals** from scratch — paged KV cache, continuous batching — to
   understand what makes serving fast.
3. **Compress to an SLM** — QLoRA + knowledge distillation (teacher → student), then
   quantize and serve the student.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full spec and
[`docs/PLAN.md`](docs/PLAN.md) for the week-by-week plan.

## Architecture

Every axis that varies — serving backend, quantizer, KV cache, trainer,
distillation loss, evaluator — sits behind a small **interface** in
[`slimserve/core/interfaces.py`](slimserve/core/interfaces.py). Concrete
implementations are **Strategies** wired in by name through a
[registry](slimserve/core/registry.py), so experiments stay declarative
(a YAML picks the components) and the code stays open for extension.

Design principles: **SOLID** throughout, with **Strategy**, **Adapter**,
**Factory/Registry**, and **Template Method** patterns where they earn their place.

```
slimserve/
├── core/          # interfaces (ABCs), typed configs, registry
├── engines/       # vLLM / HF adapters + from-scratch mini_engine (Phase 2)
├── quantization/  # AWQ / GPTQ / bitsandbytes
├── training/      # QLoRA + distillation strategies (Phase 3)
├── evaluation/    # BFCL tool-calling metrics
└── benchmark/     # runner + append-only results store (the hero table)
```

## Quickstart

```bash
pip install -e .            # base install (light)
pip install -e ".[serve]"   # add vLLM + AWQ when you reach Phase 1
python -m scripts.run_benchmark --config configs/teacher_fp16.yaml
```

The hero benchmark table lives in `results/benchmarks.csv`.

## Status

Scaffold + spec complete. Implementation follows `docs/PLAN.md`, Phase 1 → 3.
