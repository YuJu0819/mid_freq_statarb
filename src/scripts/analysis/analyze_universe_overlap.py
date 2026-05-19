"""
Analyze overlap between consecutive rolling universe epochs.

Prints:
  - Symbol count per epoch
  - Added / removed coins between consecutive epochs
  - Overlap count and Jaccard similarity
  - A heatmap saved to reports/universe_overlap.png

Usage:
    python -m src.scripts.analyze_universe_overlap
"""

import os
import yaml
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ...data.rolling_universe import RollingUniverse

_SNAPSHOTS_DIR = "./data/universe_snapshots"
_REPORT_DIR = "./reports"


def load_all_snapshots() -> list[dict]:
    """Load every snapshot_*.yaml sorted by date."""
    ru = RollingUniverse(_SNAPSHOTS_DIR)
    snapshots = []
    for snap_date in ru.list_snapshots():
        symbols = ru._load(snap_date)
        snapshots.append({"date": snap_date, "symbols": set(symbols)})
    return snapshots


def print_epoch_summary(snapshots: list[dict]):
    print("\n" + "=" * 60)
    print("  EPOCH SUMMARY")
    print("=" * 60)
    for i, s in enumerate(snapshots):
        label = f"Epoch {i+1}  ({s['date']})"
        print(f"  {label:<35}  {len(s['symbols']):>3} symbols")


def print_consecutive_overlap(snapshots: list[dict]):
    print("\n" + "=" * 60)
    print("  CONSECUTIVE EPOCH TRANSITIONS")
    print("=" * 60)
    for i in range(len(snapshots) - 1):
        a, b = snapshots[i], snapshots[i + 1]
        overlap = a["symbols"] & b["symbols"]
        added   = b["symbols"] - a["symbols"]
        removed = a["symbols"] - b["symbols"]
        jaccard = len(overlap) / len(a["symbols"] | b["symbols"])

        print(f"\n  {a['date']}  →  {b['date']}")
        print(f"    Overlap  : {len(overlap):>3}  ({jaccard:.1%} Jaccard)")
        print(f"    Added    : {len(added):>3}  {sorted(added)}")
        print(f"    Removed  : {len(removed):>3}  {sorted(removed)}")


def build_overlap_matrix(snapshots: list[dict]) -> pd.DataFrame:
    """N×N matrix of overlap counts between every pair of snapshots."""
    dates = [s["date"] for s in snapshots]
    n = len(snapshots)
    matrix = np.zeros((n, n), dtype=int)
    for i, j in itertools.product(range(n), repeat=2):
        matrix[i, j] = len(snapshots[i]["symbols"] & snapshots[j]["symbols"])
    return pd.DataFrame(matrix, index=dates, columns=dates)


def build_jaccard_matrix(snapshots: list[dict]) -> pd.DataFrame:
    """N×N matrix of Jaccard similarity between every pair of snapshots."""
    dates = [s["date"] for s in snapshots]
    n = len(snapshots)
    matrix = np.zeros((n, n))
    for i, j in itertools.product(range(n), repeat=2):
        union = len(snapshots[i]["symbols"] | snapshots[j]["symbols"])
        inter = len(snapshots[i]["symbols"] & snapshots[j]["symbols"])
        matrix[i, j] = inter / union if union > 0 else 0.0
    return pd.DataFrame(matrix, index=dates, columns=dates)


def plot_heatmaps(overlap_df: pd.DataFrame, jaccard_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title, fmt, cmap in [
        (axes[0], overlap_df, "Overlap Count",     "d",    "YlOrRd"),
        (axes[1], jaccard_df, "Jaccard Similarity", ".2f",  "YlGn"),
    ]:
        im = ax.imshow(df.values, cmap=cmap, vmin=0,
                       vmax=df.values.max() if title == "Overlap Count" else 1.0)
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(df.index, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                val = df.values[i, j]
                text = f"{val:{fmt}}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if val > df.values.max() * 0.6 else "black")

    fig.suptitle("Rolling Universe — Epoch Overlap Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "universe_overlap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nHeatmap saved → {path}")


def print_symbol_lifetime(snapshots: list[dict]):
    """Show how many epochs each symbol appears in."""
    counter: dict[str, int] = {}
    for s in snapshots:
        for sym in s["symbols"]:
            counter[sym] = counter.get(sym, 0) + 1

    total_epochs = len(snapshots)
    lifetime_df = (
        pd.Series(counter)
        .rename("epochs_present")
        .to_frame()
        .assign(fraction=lambda d: d["epochs_present"] / total_epochs)
        .sort_values("epochs_present", ascending=False)
    )

    print("\n" + "=" * 60)
    print("  SYMBOL LIFETIME DISTRIBUTION")
    print("=" * 60)
    dist = lifetime_df["epochs_present"].value_counts().sort_index()
    for epochs_count, n_symbols in dist.items():
        bar = "█" * n_symbols
        print(f"  Present in {epochs_count}/{total_epochs} epochs : {n_symbols:>3} symbols  {bar}")

    print(f"\n  Always present ({total_epochs}/{total_epochs}):")
    always = lifetime_df[lifetime_df["epochs_present"] == total_epochs].index.tolist()
    print(f"    {sorted(always)}")

    print(f"\n  Never repeated (appeared in exactly 1 epoch):")
    once = lifetime_df[lifetime_df["epochs_present"] == 1].index.tolist()
    print(f"    {sorted(once)}")


def main():
    snapshots = load_all_snapshots()
    if len(snapshots) < 2:
        print("Need at least 2 snapshots for overlap analysis.")
        return

    print_epoch_summary(snapshots)
    print_consecutive_overlap(snapshots)

    overlap_df = build_overlap_matrix(snapshots)
    jaccard_df = build_jaccard_matrix(snapshots)

    print("\n" + "=" * 60)
    print("  FULL OVERLAP COUNT MATRIX")
    print("=" * 60)
    print(overlap_df.to_string())

    print("\n" + "=" * 60)
    print("  FULL JACCARD SIMILARITY MATRIX")
    print("=" * 60)
    print(jaccard_df.round(3).to_string())

    print_symbol_lifetime(snapshots)
    plot_heatmaps(overlap_df, jaccard_df, _REPORT_DIR)


if __name__ == "__main__":
    main()
