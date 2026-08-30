"""Generate the SlimServe cost-vs-quality chart from results/benchmarks.csv.

Two panels (small multiples), NOT a dual-axis chart: cost and accuracy live on
totally different scales. Colors are the colorblind-safe Okabe-Ito set and every
bar is directly labeled, so identity never rests on color alone.

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

# fixed order: 7B teacher variants, then each student as a trio (base -> fine-tuned)
ORDER = ["teacher_fp16", "teacher_int8", "teacher_int4_awq",
         "student_1p5b_base", "student_1p5b_gold", "student_1p5b_distill",
         "student_0p5b_base", "student_0p5b_gold", "student_0p5b_distill"]
LABELS = {
    "teacher_fp16": "7B\nfp16", "teacher_int8": "7B\nint8",
    "teacher_int4_awq": "7B\nint4", "student_1p5b_base": "1.5B\nbase",
    "student_1p5b_gold": "1.5B\ngold", "student_1p5b_distill": "1.5B\ndistil",
    "student_0p5b_base": "0.5B\nbase", "student_0p5b_gold": "0.5B\ngold",
    "student_0p5b_distill": "0.5B\ndistil",
}
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"


def main() -> None:
    df = pd.read_csv(CSV).set_index("config_name").loc[ORDER]
    x = range(len(ORDER))
    xlabels = [LABELS[c] for c in ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.0))

    # --- Panel 1: cost ---
    bars = ax1.bar(x, df["cost_per_1m_tokens"].values, color=GREEN, width=0.66, zorder=3)
    ax1.set_title("Serving cost  ($ / 1M tokens)", fontsize=11.5, fontweight="bold",
                  loc="left", pad=10, color="#222")
    ax1.set_ylim(0, df["cost_per_1m_tokens"].max() * 1.18)
    for b, v in zip(bars, df["cost_per_1m_tokens"].values):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(), f"${v:.3f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#222")

    # --- Panel 2: accuracy (tool + arg, grouped) ---
    w = 0.36
    x0 = [i - w / 2 for i in x]
    x1 = [i + w / 2 for i in x]
    b_tool = ax2.bar(x0, df["tool_acc"].values, width=w, color=BLUE, zorder=3, label="tool acc")
    b_arg = ax2.bar(x1, df["arg_acc"].values, width=w, color=ORANGE, zorder=3, label="arg acc")
    ax2.set_title("Tool-calling accuracy", fontsize=11.5, fontweight="bold",
                  loc="left", pad=10, color="#222")
    ax2.set_ylim(0, 1.28)
    for bars_, vals in ((b_tool, df["tool_acc"].values), (b_arg, df["arg_acc"].values)):
        for b, v in zip(bars_, vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8, color="#333")
    ax2.legend(loc="upper center", frameon=False, fontsize=9, ncol=2)

    for ax in (ax1, ax2):
        ax.set_xticks(list(x))
        ax.set_xticklabels(xlabels, fontsize=9.5)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(length=0, labelcolor="#444")
        ax.set_yticks([])

    fig.suptitle(
        "A fine-tuned 1.5B matches the 7B teacher on tool-calling — at ~4.5× lower cost",
        fontsize=12.5, fontweight="bold", x=0.015, ha="left", color="#111")
    fig.text(0.015, 0.005,
             "Qwen2.5 on Kaggle T4s  ·  accuracy on 200 held-out xLAM tool calls  ·  "
             "gold = SFT on labels, distil = distilled from the 7B",
             fontsize=8.5, color="#888", ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
