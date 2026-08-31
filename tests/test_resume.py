"""Tests for resumability helpers — checkpoint discovery + results de-dup."""
from slimserve.benchmark.results_store import ResultsStore
from slimserve.evaluation.metrics import BenchmarkResult
from slimserve.training.checkpoints import latest_checkpoint


def _result(name):
    return BenchmarkResult(config_name=name, params_b=1.5, precision="fp16",
                           tool_acc=1.0, arg_acc=0.8, tokens_per_s=100.0,
                           ttft_ms=10.0, p99_latency_ms=50.0, vram_mb=1000.0,
                           cost_per_1m_tokens=0.05)


def test_latest_checkpoint_none_when_absent_or_empty(tmp_path):
    assert latest_checkpoint(str(tmp_path / "missing")) is None
    assert latest_checkpoint(str(tmp_path)) is None          # exists but empty


def test_latest_checkpoint_picks_highest_step(tmp_path):
    for n in (50, 200, 100):
        (tmp_path / f"checkpoint-{n}").mkdir()
    (tmp_path / "checkpoint-notanumber").mkdir()             # ignored
    (tmp_path / "adapter").mkdir()                           # ignored
    assert latest_checkpoint(str(tmp_path)).endswith("checkpoint-200")


def test_results_store_has(tmp_path):
    store = ResultsStore(str(tmp_path / "b.csv"))
    assert store.has("teacher_fp16") is False               # no file yet
    store.append(_result("teacher_fp16"))
    assert store.has("teacher_fp16") is True
    assert store.has("student_0p5b_gold") is False
