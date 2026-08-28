"""Generate the Phase 1 cost-vs-quality chart from results/benchmarks.csv.

Small-multiple bar panels (magnitude comparison) — deliberately NOT a dual-axis
chart, since cost (~0.1) and accuracy (~0.99) live on totally different scales.
Colors are the colorblind-safe Okabe-Ito set, and every bar is directly labeled
so identity never depends on color alone.

    python scripts/plot_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

CSV = "results/benchmarks.csv"
OUT = "results/charts/phase1_cost_vs_quality.png"

ORDER = ["fp16", "int8", "int4"]                       # fixed category order
COLORS = {"fp16": "#0072B2", "int8": "#E69F00", "int4": "#009E73"}
LABELS = {"fp16": "FP16", "int8": "INT8", "int4": "INT4"}


def main() -> None:
    df = pd.read_csv(CSV).set_index("precision").loc[ORDER]
    x = range(len(ORDER))
    colors = [COLORS[p] for p in ORDER]
    xlabels = [LABELS[p] for p in ORDER]

    panels = [
        ("Cost  ($ / 1M tokens)", df["cost_per_1m_tokens"], "${:.3f}", None),
        ("Tool-calling accuracy", df["tool_acc"], "{:.2f}", (0, 1.08)),
        ("Peak VRAM  (GB)", df["vram_mb"] / 1000.0, "{:.1f}", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, (title, series, fmt, ylim) in zip(axes, panels):
        bars = ax.bar(x, series.values, color=colors, width=0.62, zorder=3)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10, loc="left",
                     color="#222")
        ax.set_xticks(list(x))
        ax.set_xticklabels(xlabels, fontsize=10)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, series.max() * 1.18)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(length=0, labelcolor="#444444")
        ax.set_yticks([])
        for b, v in zip(bars, series.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color="#222222")

    fig.suptitle(
        "Quantizing Qwen2.5-7B for tool calling — cost & memory drop, accuracy holds",
        fontsize=12.5, fontweight="bold", x=0.015, ha="left", color="#111111")
    fig.text(
        0.015, 0.01,
        "7B teacher on Kaggle T4s  ·  FP16 = 2×T4, INT8/INT4 = 1×T4  ·  "
        "accuracy on 200 held-out xLAM tool calls",
        fontsize=8.5, color="#888888", ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
