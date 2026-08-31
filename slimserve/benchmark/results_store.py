"""Append-only store for benchmark rows (results/benchmarks.csv).

Every run writes here immediately so the hero table is never lost to a crashed
notebook session. Plain CSV so it is diff-able and readable in the repo.
"""
from __future__ import annotations

import csv
from pathlib import Path

from slimserve.evaluation.metrics import BenchmarkResult


class ResultsStore:
    def __init__(self, path: str = "results/benchmarks.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: BenchmarkResult) -> None:
        row = result.as_row()
        new_file = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def has(self, config_name: str) -> bool:
        """Whether a row for ``config_name`` is already recorded.

        Lets a benchmark suite resume after a kill by skipping done configs (and
        avoids appending a duplicate row when a single config is re-run).
        """
        if not self.path.exists():
            return False
        with self.path.open(newline="") as f:
            return any(r["config_name"] == config_name for r in csv.DictReader(f))
