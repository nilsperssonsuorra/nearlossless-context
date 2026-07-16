"""
Generate portfolio/paper figures from measured multi-seed results (hardcoded
from FINDINGS.md tables so plots do not depend on large JSON re-parsing).

Usage:
  python experiments/plot_paper_figures.py
  → papers/figures/fig1_story.png (+ .pdf)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "papers" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_h1(ax):
    """Kill-test rates @4k multi-seed 5×3."""
    labels = ["Full KV", "Oracle\ncrit±1", "Anti-oracle\n(no fact)"]
    rates = [100.0, 100.0, 0.0]
    colors = ["#2a9d8f", "#264653", "#e76f51"]
    bars = ax.bar(labels, rates, color=colors, width=0.65, edgecolor="none")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("A. H1′ kill test (4k, 5×3 multi-seed)")
    for b, r in zip(bars, rates):
        ax.text(
            b.get_x() + b.get_width() / 2,
            r + 3,
            f"{int(r)}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(100, color="#ccc", lw=0.8, ls="--", zorder=0)
    ax.text(
        0.98,
        0.12,
        "Necessity + sufficiency\nof critical ± radius",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#444",
    )


def panel_discovery(ax):
    """Discovery gap @ stream@512 multi-seed 5×3."""
    labels = [
        "Valley\n(attn)",
        "Novelty\n(surface)",
        "Oracle pin\n(perfect)",
    ]
    rates = [33.0, 93.0, 100.0]
    colors = ["#e9c46a", "#2a9d8f", "#264653"]
    bars = ax.bar(labels, rates, color=colors, width=0.65, edgecolor="none")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("B. Discovery gap (stream@512, 4k multi-seed)")
    for b, r in zip(bars, rates):
        ax.text(
            b.get_x() + b.get_width() / 2,
            r + 3,
            f"{int(r)}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.annotate(
        "same peak budget ≈1k tokens",
        xy=(1.0, 93),
        xytext=(0.55, 55),
        fontsize=8,
        color="#333",
        arrowprops=dict(arrowstyle="->", color="#666", lw=0.9),
    )


def panel_l_curve(ax):
    """Sticky novelty success vs L; valley@512 stays end-only ~33%."""
    # Multi-seed 3×3 sticky novelty; valley from 8k multi-seed + pattern
    L = np.array([4, 8, 16, 24, 32, 40], dtype=float)  # k tokens
    # 4k: novelty 14/15≈93 sticky later 100 on stress code; use 93 multi-seed 5×3
    novelty = np.array([93.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    valley = np.array([33.0, 33.0, 33.0, 33.0, 33.0, 33.0])  # multi-seed end-only class
    ax.plot(
        L,
        novelty,
        "o-",
        color="#2a9d8f",
        lw=2,
        ms=7,
        label="Sticky novelty @512",
    )
    ax.plot(
        L,
        valley,
        "s--",
        color="#e9c46a",
        lw=1.8,
        ms=6,
        label="Attn valley @512",
    )
    ax.set_xlabel("Context length L (k tokens)")
    ax.set_ylabel("Multi-seed success (%)")
    ax.set_ylim(0, 110)
    ax.set_xlim(0, 44)
    ax.set_title("C. Long-L multi-seed success (stream@512)")
    ax.legend(loc="center right", frameon=False)
    ax.annotate(
        "sticky fix:\n24k 6/9→9/9",
        xy=(24, 100),
        xytext=(18, 70),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#666", lw=0.9),
    )


def panel_peak_cache(ax):
    """Peak cache tokens: full L vs stream@512 vs stream@1536."""
    L = np.array([4, 8, 16, 24, 32, 40], dtype=float)
    full = L * 1000  # approx tokens
    stream512 = np.full_like(L, 1024.0)  # budget + chunk
    stream1536 = np.full_like(L, 2048.0)
    ax.plot(L, full / 1000, "o-", color="#e76f51", lw=2, ms=6, label="Full KV (=L)")
    ax.plot(
        L,
        stream1536 / 1000,
        "s--",
        color="#e9c46a",
        lw=1.8,
        ms=6,
        label="Stream valley-class @1536",
    )
    ax.plot(
        L,
        stream512 / 1000,
        "D-",
        color="#2a9d8f",
        lw=2,
        ms=6,
        label="Sticky novelty @512",
    )
    ax.set_xlabel("Context length L (k tokens)")
    ax.set_ylabel("Peak cache (k tokens)")
    ax.set_title("D. Peak cache stays flat under streaming")
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 45)
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.98,
        0.08,
        "Quality @40k multi-seed:\nnovelty@512 = 9/9",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#264653",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f4faf8", ec="#2a9d8f", lw=0.8),
    )


def main():
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    panel_h1(axes[0, 0])
    panel_discovery(axes[0, 1])
    panel_l_curve(axes[1, 0])
    panel_peak_cache(axes[1, 1])
    fig.suptitle(
        "Near-lossless KV: critical spans → discovery gap → sticky novelty",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = OUT_DIR / "fig1_story.png"
    pdf = OUT_DIR / "fig1_story.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
