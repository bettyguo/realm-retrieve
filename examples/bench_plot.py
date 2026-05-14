#!/usr/bin/env python3
"""Render the accuracy-vs-retrieval-calls figure from the paper.

Saves ``assets/results.svg``. Uses only matplotlib (no external services).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Numbers from Table 1 of the paper. Keep in sync with README.md.
POINTS = {
    "No Retrieval":   (0.0, 48.7, "#888888"),
    "Single RAG":     (1.0, 59.4, "#1f77b4"),
    "IRCoT":          (3.4, 65.4, "#ff7f0e"),
    "FLARE":          (2.8, 62.3, "#2ca02c"),
    "Self-RAG":       (2.1, 61.9, "#d62728"),
    "Search-R1":      (2.4, 66.8, "#9467bd"),
    "ReaLM-Retrieve": (1.8, 71.2, "#e63946"),
}


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=120)
    for name, (x, y, color) in POINTS.items():
        size = 240 if name == "ReaLM-Retrieve" else 110
        ax.scatter(x, y, s=size, c=color, edgecolors="black", linewidth=0.8, zorder=3)
        dx, dy = (0.05, 0.6) if name != "No Retrieval" else (0.05, -1.4)
        weight = "bold" if name == "ReaLM-Retrieve" else "normal"
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dx, y + dy),
            fontsize=9,
            fontweight=weight,
        )

    ax.set_xlabel("Retrieval calls per question  (↓ better)")
    ax.set_ylabel("F1 on MuSiQue  (↑ better)")
    ax.set_title("ReaLM-Retrieve dominates the accuracy-efficiency Pareto front")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-0.3, 4.0)
    ax.set_ylim(45, 75)

    out_svg = out_dir / "results.svg"
    out_png = out_dir / "results.png"
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_png, format="png")
    print(f"Wrote {out_svg.relative_to(out_dir.parent)}")
    print(f"Wrote {out_png.relative_to(out_dir.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
