# SlimServe

**Compress a large tool-calling LLM into a fast, cheap SLM — and prove the cost delta at every step.**

Big models are expensive to serve. For a narrow, high-value task —
**agentic tool / function calling** — a well-compressed small model can match a
large one at a fraction of the cost. SlimServe takes a 7B teacher, shrinks it to
a ~1.5B student, and measures **$/1M tokens, throughput, latency, VRAM, and
tool-calling accuracy** across every configuration.

> **Headline target:** distilled INT4 SLM at **~6–8× lower cost**, **≥~90–95% of
> teacher tool-calling accuracy**.

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
