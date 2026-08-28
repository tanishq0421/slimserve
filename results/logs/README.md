# Run logs

Two kinds of evidence for every benchmark run:

- **Curated summaries** (`phase1_*.md`) — the readable version: command, hardware,
  key numbers, memory breakdown. What you actually want to *read*.
- **Raw console dumps** (`raw/*.log`) — verbatim, unedited stdout+stderr from the
  run. Kept as **proof the run really happened on a GPU**: they contain the vLLM
  engine init, the CUDA device / NCCL lines, CUDA-graph capture, and real
  timestamps that can't be faked convincingly by hand.

## Capturing a raw log

`tee` the run so the console output is saved verbatim while you still watch it live:

```bash
python -m scripts.run_benchmark --config configs/teacher_int4_awq.yaml \
  2>&1 | tee results/logs/raw/teacher_int4_awq.log
```

## Strongest proof: the Kaggle notebook itself

Even better than a text log, **Save a Version** of the Kaggle notebook (and optionally
make it public). The saved notebook preserves the code, the GPU accelerator setting,
and the full run output together — link it from the top-level README. A public
notebook is the most credible "this really ran on a T4" evidence there is.
