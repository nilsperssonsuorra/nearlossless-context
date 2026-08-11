"""Generate Figure 1 from checked-in machine-readable paper aggregates."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "papers" / "figures"
AGGREGATES = ROOT / "results" / "paper_aggregates.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rows(group: str) -> list[dict[str, str]]:
    with AGGREGATES.open(encoding="utf-8", newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["claim_group"] == group]


def _rate(group: str, arm: str, context: int) -> float:
    row = next(
        r
        for r in _rows(group)
        if r["arm"] == arm and int(r["context_tokens"]) == context
    )
    return 100.0 * float(row["success_rate"])


def _style() -> None:
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


def _bar_labels(ax, bars, rates) -> None:
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 3,
            f"{rate:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def panel_h1(ax) -> None:
    labels = ["Full KV", "Oracle\ncrit ±1", "Anti-oracle\n(no fact)"]
    rates = [
        _rate("h1", "full", 4096),
        _rate("h1", "oracle_r1", 4096),
        _rate("h1", "anti_oracle", 4096),
    ]
    bars = ax.bar(
        labels,
        rates,
        color=["#2a9d8f", "#264653", "#e76f51"],
        width=0.65,
        edgecolor="none",
    )
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("A. Kill test (4k, 5x3 multi-seed)")
    _bar_labels(ax, bars, rates)
    ax.axhline(100, color="#ccc", lw=0.8, ls="--", zorder=0)
    ax.text(
        0.98,
        0.12,
        "Necessary and sufficient\nwithin this protocol",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#444",
    )


def panel_discovery(ax) -> None:
    labels = ["Valley\n(attn)", "Novelty\n(surface)", "Oracle pin\n(perfect)"]
    rates = [
        _rate("discovery", "stream_valley", 4096),
        _rate("discovery", "stream_novelty", 4096),
        _rate("h1", "oracle_r1", 4096),
    ]
    bars = ax.bar(
        labels,
        rates,
        color=["#e9c46a", "#2a9d8f", "#264653"],
        width=0.65,
        edgecolor="none",
    )
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("B. Discovery gap (stream@512, 4k)")
    _bar_labels(ax, bars, rates)
    ax.annotate(
        "Same peak cache: ~1k tokens",
        xy=(1.0, rates[1]),
        xytext=(0.42, 59),
        fontsize=9.5,
        fontweight="bold",
        color="#1f2933",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#4b5563", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#4b5563", lw=1.2),
    )


def panel_l_curve(ax) -> None:
    rows = _rows("long_context")
    novelty_rows = sorted(
        (r for r in rows if r["arm"] == "stream_novelty"),
        key=lambda r: int(r["context_tokens"]),
    )
    valley_rows = sorted(
        (r for r in rows if r["arm"] == "stream_valley"),
        key=lambda r: int(r["context_tokens"]),
    )
    novelty_l = np.array([int(r["context_tokens"]) / 1024 for r in novelty_rows])
    novelty = np.array([100 * float(r["success_rate"]) for r in novelty_rows])
    valley_l = np.array([int(r["context_tokens"]) / 1024 for r in valley_rows])
    valley = np.array([100 * float(r["success_rate"]) for r in valley_rows])
    ax.plot(
        novelty_l,
        novelty,
        "o-",
        color="#2a9d8f",
        lw=2,
        ms=7,
        label="Sticky novelty @512",
    )
    ax.plot(
        valley_l,
        valley,
        "s--",
        color="#e9c46a",
        lw=1.8,
        ms=6,
        label="Attn valley @512 (measured)",
    )
    ax.set_xlabel("Context length L (k tokens)")
    ax.set_ylabel("Multi-seed success (%)")
    ax.set_ylim(0, 110)
    ax.set_xlim(0, 44)
    ax.set_title("C. Long-context measured cells (stream@512)")
    ax.legend(loc="center right", frameon=False)


def panel_peak_cache(ax) -> None:
    rows = _rows("systems")
    full_rows = sorted(
        (r for r in rows if r["arm"] == "full"),
        key=lambda r: int(r["context_tokens"]),
    )
    novelty_rows = sorted(
        (r for r in rows if r["arm"] == "stream_novelty"),
        key=lambda r: int(r["context_tokens"]),
    )
    full_l = np.array([int(r["context_tokens"]) / 1024 for r in full_rows])
    full_peak = np.array([float(r["mean_peak_cache"]) / 1024 for r in full_rows])
    novelty_l = np.array([int(r["context_tokens"]) / 1024 for r in novelty_rows])
    novelty_peak = np.array(
        [float(r["mean_peak_cache"]) / 1024 for r in novelty_rows]
    )
    ax.plot(full_l, full_peak, "o-", color="#e76f51", lw=2, ms=6, label="Full KV")
    ax.plot(
        novelty_l,
        novelty_peak,
        "D-",
        color="#2a9d8f",
        lw=2,
        ms=6,
        label="Sticky novelty @512",
    )
    ax.set_xlabel("Context length L (k tokens)")
    ax.set_ylabel("Peak cache (k tokens)")
    ax.set_title("D. Measured peak cache")
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


def main() -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    panel_h1(axes[0, 0])
    panel_discovery(axes[0, 1])
    panel_l_curve(axes[1, 0])
    panel_peak_cache(axes[1, 1])
    fig.suptitle(
        "Critical spans, the discovery gap, and sticky novelty",
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
