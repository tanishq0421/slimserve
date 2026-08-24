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
