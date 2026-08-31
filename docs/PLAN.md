# SlimServe — Week-by-Week Execution Plan

Incremental by design: **use the tools → rebuild them → push to a novel result.**
Each week ends with a concrete, committable deliverable. Learn concepts *just before*
you need them (links below), not upfront.

Assumes ~10–15 hrs/week on **Kaggle** (2× T4). Adjust freely.

---

## Phase 1 — Serve & Benchmark (Weeks 1–2)

### Week 1 — Baseline serving
- [ ] Set up Kaggle GPU notebook; load `Qwen2.5-7B-Instruct`.
- [ ] Serve with **vLLM**; send tool-calling prompts, confirm valid tool-call output.
- [ ] Write `serving/benchmark.py`: measure **tok/s, TTFT, p50/p99 latency, peak VRAM**
      under a fixed request load.
- [ ] Record the **FP16 baseline row** in `results/benchmarks.csv`.
- **Learn:** what the KV cache is and why it dominates memory; what TTFT vs throughput mean.
- **Deliverable:** baseline benchmark row + a working load-test script.

### Week 2 — Quantize & KV-cache compression
- [ ] Quantize teacher to **INT8**, then **INT4 (AWQ or GPTQ)**; re-benchmark each.
- [ ] Enable **KV-cache quantization** in vLLM; re-benchmark.
- [ ] Tune **continuous batching** (max-num-seqs, batch size); note throughput gains.
- [ ] Compute **$/1M tokens** for each row (fix a reference GPU price).
- **Learn:** how INT8/INT4 weight quantization works; why activation *outliers* matter
      (AWQ/SmoothQuant intuition).
- **Deliverable:** benchmark table with 4–5 rows + a README section explaining each knob.
- ✅ **Milestone: an above-average portfolio repo already exists.**

---

## Phase 2 — Compress to an SLM (Weeks 3–5)

### Week 3 — QLoRA fine-tune the student
- [ ] QLoRA-fine-tune `Qwen2.5-1.5B-Instruct` on `xlam-function-calling-60k` using
      **PEFT + Unsloth** on a single T4.
- [ ] Evaluate on a held-out slice; add the **Student-SFT row**.
- **Learn:** LoRA (low-rank adapters) and QLoRA (NF4 base + LoRA); why it fits one GPU.
- **Deliverable:** `compress/qlora_finetune.py` + student SFT benchmark row.

### Week 4 — Distillation
- [ ] Generate teacher tool-call completions; train student on them (**sequence-level KD**).
- [ ] (Stretch) add **logit KD** (KL to teacher soft labels).
- [ ] Add the **distilled-student row**; compare vs plain SFT.
- **Learn:** knowledge distillation; soft labels vs hard labels.
- **Deliverable:** `compress/distill.py` + accuracy-retained comparison.

### Week 5 — Final SLM + the story
- [ ] Quantize the distilled student to **INT4**; serve it; add the final row.
- [ ] Set up the **BFCL** harness for the headline accuracy number.
- [ ] Generate the **cost-vs-quality Pareto chart**; write the README headline result.
- [ ] (Bonus) **speculative decoding**: student drafts, teacher verifies → latency win.
- **Deliverable:** complete benchmark table + Pareto chart + polished README.
- ✅ **Milestone: the full LLM→SLM cost-optimization story, end to end.**

---

## Phase 3 — Rebuild the Internals (Weeks 6–8)

### Week 6 — Manual inference loop
- [ ] In raw PyTorch, load a small model (e.g. `Qwen2.5-0.5B`) and write a decode loop
      by hand with a **naive contiguous KV cache**.
- [ ] Verify output matches HF `generate()`.
- **Learn:** attention at inference time; prefill vs decode; why KV cache = no recompute.
- **Deliverable:** `minigpt_engine/decode.py` + `kv_cache.py` (naive).

### Week 7 — Paged KV cache
- [ ] Implement a **simplified PagedAttention**: fixed-size KV blocks + a block table.
- [ ] Show the **memory-fragmentation win** vs contiguous cache with a small experiment.
- **Learn:** vLLM's PagedAttention paper (core idea only, not the CUDA).
- **Deliverable:** `kv_cache.py` (paged) + a fragmentation before/after chart.

### Week 8 — Toy scheduler + comparison
- [ ] Implement a minimal **continuous-batching scheduler** (requests join/leave mid-gen).
- [ ] Benchmark your engine vs vLLM; write up the gap honestly.
- **Deliverable:** `scheduler.py` + "what I learned rebuilding vLLM's core" writeup.
- ✅ **Milestone: deep systems signal — you understand what you built, not just used.**

---

## Definition of done

- `results/benchmarks.csv` has ≥6 configs; README shows the cost-vs-quality Pareto chart.
- Headline: **distilled INT4 SLM at ≥~6× lower $/1M-token cost, ≥~90–95% teacher BFCL acc.**
- Phase-3 engine generates correctly and demonstrably reduces KV-cache memory via paging.
- README is skimmable in 2 minutes with the cost story up top.

## Working notes

- **Commit in small chunks** — one deliverable per commit, no bundling.
- Keep every training run short and checkpointed (free-tier sessions time out).
- Record *every* benchmark run into `benchmarks.csv` immediately — that file is the
  source of truth for the whole project.
