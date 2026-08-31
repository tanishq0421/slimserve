# SlimServe — Design Spec

**Date:** 2026-08-24
**Author:** Tanishq (int.ankit@surgegrowth.io)
**Status:** Approved for implementation

---

## 1. One-line pitch

Compress a large tool-calling model into a fast, cheap **Small Language Model (SLM)**
that does **agentic tool / function calling** as well as its teacher — and prove the
**cost/quality delta at every step** with a single benchmark table.

## 2. Why this project (production relevance)

The 2026 production consensus is that for narrow, schema-constrained, tool-calling-heavy
agent work, a well-fine-tuned 3B–7B model is faster, cheaper, more predictable, and
easier to evaluate than a frontier LLM — often at **~5–20× lower cost** for comparable
task quality. The industry-standard compression recipe is **P-KD-Q: Prune → Distill →
Quantize**, which is exactly this project's spine.

This repo demonstrates the full **inference-cost engineering** skillset an ML-infra /
inference-engineer role screens for:

- Serving at scale (vLLM: paged KV cache, continuous batching)
- Quantization (INT8 / INT4 / FP8, plus KV-cache quantization)
- KV-cache internals (paged attention, from scratch)
- LoRA / QLoRA fine-tuning
- Knowledge distillation (LLM → SLM)
- Rigorous cost/latency/throughput/quality benchmarking

## 3. The task: agentic tool-calling

The model takes a user request + a set of available tool schemas and must emit the
**correct tool call(s)** with valid arguments (structured JSON). This is:

- **Highly in-demand** — "SLM-first agents" is the dominant 2026 pattern; a Dec-2025
  Amazon Science paper shows fine-tuned SLMs *beat* large models on tool calling.
- **Exactly measurable** — tool-selection accuracy, valid-schema rate, argument
  correctness, on standard benchmarks (BFCL / ToolBench). No fuzzy human grading.
- **A superset of structured extraction** — keeps clean metrics while riding the
  agents narrative that hiring teams care about.

## 4. Models

| Role | Model | Why |
|---|---|---|
| **Teacher** | `Qwen2.5-7B-Instruct` | Strong native tool-calling; same family as student = cleaner distillation. (Alt: `Llama-3.1-8B-Instruct`.) |
| **Student** | `Qwen2.5-1.5B-Instruct` (also try `3B`) | Small enough to be genuinely cheap; big enough to learn tool-calling well. |

Same tokenizer/family between teacher and student removes a class of distillation
headaches (vocab mismatch), so we start there.

## 5. Datasets

- **Training (fine-tune / distill):** `Salesforce/xlam-function-calling-60k` and/or
  `glaive-function-calling-v2` — large, high-quality tool-call datasets.
- **Evaluation:** **Berkeley Function-Calling Leaderboard (BFCL)** harness — the
  standard, with categories for simple / multiple / parallel / irrelevance detection.
  (Secondary: ToolBench pass-rate.)

## 6. The hero artifact: one benchmark table

Every configuration adds a row. This table *is* the project.

| Config | Params | Precision | BFCL acc | tok/s | TTFT (ms) | p99 lat | VRAM | $/1M tok |
|---|---|---|---|---|---|---|---|---|
| Teacher FP16 (baseline) | 7B | FP16 | … | … | … | … | … | … |
| Teacher INT8 | 7B | INT8 | … | … | … | … | … | … |
| Teacher INT4 (AWQ) | 7B | INT4 | … | … | … | … | … | … |
| Teacher INT4 + KV-quant | 7B | INT4 | … | … | … | … | … | … |
| Student SFT FP16 | 1.5B | FP16 | … | … | … | … | … | … |
| Student distilled FP16 | 1.5B | FP16 | … | … | … | … | … | … |
| **Student distilled INT4** | **1.5B** | **INT4** | … | … | … | … | … | … |

**Cost model:** `$/1M tokens = (GPU $/hr) ÷ (throughput tok/hr)`, using a fixed
reference GPU price (state the GPU + rate). Report the **cost-vs-quality Pareto chart**
as the writeup centerpiece.

## 7. Architecture (OOP, design-principled)

**Core idea:** every axis that *varies* in this project — how a model is served,
quantized, cached, trained, distilled, evaluated — is a **Strategy** behind a small
abstract interface (`slimserve/core/interfaces.py`). Concrete implementations register
themselves by name in a **Factory/Registry**, so experiments stay declarative (a YAML
config names the components) and the code is **open for extension, closed for
modification**. Comparing swappable components *is* the project's whole point, so this
structure is the right fit — not abstraction for its own sake.

```
slimserve/                       # repo root
├── README.md                    # pitch + headline results + charts
├── pyproject.toml               # package + phase-scoped optional deps
├── docs/{DESIGN.md, PLAN.md}
├── configs/                     # one YAML per experiment (declarative)
├── scripts/                     # thin CLI entrypoints (argparse -> pipeline)
├── results/{benchmarks.csv, charts/}   # the hero table (source of truth)
└── slimserve/                   # the package
    ├── core/
    │   ├── interfaces.py        # ABCs: InferenceEngine, Quantizer, KVCache,
    │   │                        #   Scheduler, Trainer, DistillStrategy, Evaluator
    │   ├── config.py            # frozen dataclasses (DTOs + stage configs)
    │   └── registry.py          # name -> class registry (Factory)
    ├── engines/                 # InferenceEngine implementations
    │   ├── vllm_engine.py       #   Adapter around vLLM (Phase 1)
    │   ├── hf_engine.py         #   Adapter around HuggingFace generate()
    │   └── mini_engine/         #   Phase 3: from-scratch engine
    │       ├── kv_cache.py      #     Contiguous / Paged KV cache
    │       └── scheduler.py     #     continuous-batching scheduler
    ├── quantization/            # Quantizer: awq.py, gptq.py, bnb.py
    ├── training/                # Trainer (Template Method base loop)
    │   ├── base.py  qlora_trainer.py
    │   └── distillation/        # DistillStrategy: sequence_kd.py, logit_kd.py
    ├── evaluation/              # Evaluator: bfcl_evaluator.py + metrics.py
    ├── benchmark/               # runner.py (orchestrator) + results_store.py
    └── pipeline.py              # wires stages via registry + interfaces
```

Each module has one clear job, a small interface, and is runnable standalone.

### Design principles applied

| Principle / pattern | Where |
|---|---|
| **SRP** | one module = one job (a quantizer only quantizes) |
| **OCP** | add an engine/quantizer/loss without editing existing code |
| **LSP** | `BenchmarkRunner` works with *any* `InferenceEngine` |
| **ISP** | interfaces are tiny (2–4 methods), no fat base classes |
| **DIP** | runner/pipeline depend on ABCs, not concrete libs |
| **Strategy** | engines, quantizers, distill losses, evaluators |
| **Adapter** | `VLLMEngine`/`HFEngine` wrap third-party APIs into our interface |
| **Factory + Registry** | `build("engine", "vllm", cfg)` from a config name |
| **Template Method** | `BaseTrainer` loop; subclasses override `compute_loss` |

**Guardrail:** keep Phase 3's `mini_engine/` readable over clever — its job is to
teach. Don't add an interface until a second implementation needs it (YAGNI).

## 8. Phases & deliverables

### Phase 1 — Serve & benchmark with existing tools
- Serve teacher on **vLLM**; establish the FP16 baseline row.
- Quantize INT8 → INT4 (**AWQ**/GPTQ) + **KV-cache quantization**; tune continuous batching.
- **Deliverable:** populated benchmark table + README explaining every knob.

### Phase 2 — Compress to a true SLM
- **QLoRA**-fine-tune the student on tool-calling data (PEFT + Unsloth, single T4).
- **Distill** teacher → student: sequence-level KD (train on teacher completions) +
  optional logit KD. Measure accuracy retained.
- Quantize the student to INT4; serve it. Bonus: **speculative decoding** (student
  drafts, teacher verifies).
- **Deliverable:** the LLM→SLM cost-vs-quality Pareto chart + headline result.

### Phase 3 — Rebuild the internals from scratch
- Manual KV cache + decode loop in raw PyTorch for a small model.
- Simplified **paged KV cache** (PagedAttention) → show fragmentation win.
- Toy **continuous-batching scheduler**; compare vs vLLM.
- **Deliverable:** `minigpt_engine/` + a "what I learned rebuilding vLLM's core" writeup.

## 9. Success criteria

- A reproducible benchmark table with **≥6 configs** across precision + model size.
- Headline: **distilled INT4 SLM at ≥~6× lower $/1M-token cost and ≥~90–95% of teacher
  BFCL accuracy.**
- Phase-3 from-scratch engine that correctly generates and demonstrably reduces KV-cache
  memory via paging.
- A README a hiring engineer can skim in 2 minutes and grasp the cost story.

## 10. Compute & constraints

- **Primary compute:** Kaggle Notebooks (2× T4 / 32GB, ~30h/week). Local 8GB-RAM laptop
  is just the terminal.
- **Fits free tier:** 7B in INT4 inference, QLoRA on a ≤3B student, serving benchmarks.
- **FP8 caveat:** T4 (Turing) has no native FP8. For the FP8 row, rent an Ada/Hopper GPU
  (e.g. RunPod/Modal) for a short one-off run, or mark FP8 as an optional stretch row.

## 11. Explicitly out of scope (YAGNI)

- Multi-node / multi-GPU distributed serving.
- Custom CUDA kernels (Phase 3 stays in PyTorch/Triton-level clarity, not hand-CUDA).
- Building a production auth/rate-limited API gateway.
- Pruning is optional (the "P" in P-KD-Q) — include only if time allows after Phase 2.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Free-tier session timeouts interrupt long training | Checkpoint often; keep training runs < a few hours; use Unsloth for speed |
| Distilled student underperforms | Fall back to plain QLoRA SFT (still a valid, strong result); report honestly |
| BFCL harness setup friction | Start with a small held-out slice of the training set for a fast internal metric, add full BFCL after |
| FP8 unavailable on free GPU | Mark as optional stretch row (see §10) |
