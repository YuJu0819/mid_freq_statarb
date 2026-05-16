"""
HO-MoE usage analysis.

Reads the per-fold CMI tournament log, per-OOS-date regime timeline, and
per-regime expert importance CSVs produced by `train_ebm_signal.py --ho_moe`
and answers two questions:

  1. How often did each macro candidate win the CMI tournament — overall, by
     calendar quarter, and across consecutive folds?
  2. How often was each market regime active at OOS prediction time, and
     therefore how heavily was each expert used? (Regime gating is
     hysteresis-smoothed and per-OOS-date, not per fold — so this can only
     be reconstructed faithfully from `ebm_homoe_regime_timeline.csv`.)

Outputs
-------
- Console: human-readable summary tables.
- {run_dir}/ebm_homoe_usage_summary.csv : machine-readable consolidated table
  with rows like
      (kind="winner",  key="market_volatility", n=12, pct=0.27, ...)
      (kind="regime",  key="0|market_dispersion", n=80, pct=0.33, ...)
      (kind="expert_coverage", key="1", n_folds=18, mean_importance=0.42)

Usage
-----
    python -m src.scripts.analyze_homoe --run_id homoe_v1
    python -m src.scripts.analyze_homoe --run_dir ./reports/strategies/homoe_v1
    python -m src.scripts.analyze_homoe --run_id homoe_v1 --plot
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _resolve_run_dir(args: argparse.Namespace) -> str:
    if args.run_dir:
        return args.run_dir
    if args.run_id:
        return os.path.join("./reports/strategies", args.run_id)
    raise SystemExit("Must pass --run_id or --run_dir.")


def _load_cmi_log(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "ebm_homoe_cmi_log.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, parse_dates=[0])
    df = df.rename(columns={df.columns[0]: "fold_date"}).set_index("fold_date")
    return df


def _load_regime_timeline(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "ebm_homoe_regime_timeline.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, parse_dates=[0])
    df = df.rename(columns={df.columns[0]: "ts"}).set_index("ts")
    # `active_regime` may be NaN (warmup before first hysteresis-settled regime).
    df["active_regime"] = df["active_regime"].astype("string")
    return df


def _load_expert_importances(run_dir: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in glob.glob(
        os.path.join(run_dir, "ebm_expert_importance_regime_*.csv")
    ):
        m = re.search(r"regime_(.+?)\.csv$", os.path.basename(path))
        if not m:
            continue
        regime = m.group(1)
        df = pd.read_csv(path, index_col=0)
        out[regime] = df
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    bar = "─" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


def summarize_winner(cmi_log: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate winner counts + share of folds + mean raw / EMA CMI."""
    n_folds = len(cmi_log)
    counts = cmi_log["winner"].value_counts()
    raw_cols = [c for c in cmi_log.columns if c.startswith("raw_cmi_")]
    ema_cols = [c for c in cmi_log.columns if c.startswith("ema_cmi_")]
    rows = []
    for candidate in counts.index.tolist() + [
        c.replace("raw_cmi_", "") for c in raw_cols
        if c.replace("raw_cmi_", "") not in counts.index
    ]:
        raw_col = f"raw_cmi_{candidate}"
        ema_col = f"ema_cmi_{candidate}"
        rows.append({
            "candidate":        candidate,
            "n_folds_won":      int(counts.get(candidate, 0)),
            "win_share":        float(counts.get(candidate, 0) / max(n_folds, 1)),
            "mean_raw_cmi":     float(cmi_log[raw_col].mean()) if raw_col in cmi_log else np.nan,
            "mean_ema_cmi":     float(cmi_log[ema_col].mean()) if ema_col in cmi_log else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_folds_won", ascending=False)


def summarize_winner_by_quarter(cmi_log: pd.DataFrame) -> pd.DataFrame:
    """Per-quarter winner distribution — useful for spotting regime drift."""
    qkey = cmi_log.index.to_period("Q").astype(str)
    g = cmi_log.assign(quarter=qkey).groupby("quarter")["winner"]
    return g.value_counts().unstack(fill_value=0).sort_index()


def summarize_winner_streaks(cmi_log: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-fold runs of the same winner — proxy for tournament churn."""
    w = cmi_log["winner"]
    streak_id = (w != w.shift()).cumsum()
    grp = w.groupby(streak_id)
    rows = []
    for _, run in grp:
        rows.append({
            "winner":    run.iloc[0],
            "start":     run.index[0],
            "end":       run.index[-1],
            "n_folds":   int(len(run)),
        })
    return pd.DataFrame(rows)


def summarize_regime_usage(reg: pd.DataFrame) -> pd.DataFrame:
    """
    Per (winner_separator, active_regime) OOS-date counts. The "expert usage"
    of regime r under separator s equals the count of OOS days that pair was
    active — this is the column that answers "how heavily was each expert
    used?".
    """
    if reg.empty:
        return pd.DataFrame()
    df = reg.reset_index()
    grp = df.groupby(["winner_separator", "active_regime"], dropna=False)
    out = grp.size().rename("n_days").reset_index()
    out["share_overall"] = out["n_days"] / out["n_days"].sum()
    # Share within each separator block — i.e. "given we were on this
    # separator, what fraction of days was this regime active?".
    out["share_within_winner"] = out.groupby("winner_separator")["n_days"].transform(
        lambda x: x / x.sum()
    )
    out["expert_available_pct"] = grp["expert_available"].mean().values
    return out.sort_values(["winner_separator", "n_days"], ascending=[True, False])


def summarize_expert_coverage(
    expert_imps: dict[str, pd.DataFrame],
    reg: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Per-regime expert coverage: number of folds an expert was trained, mean
    of mean-feature-importance (a coarse "did this expert learn anything"
    signal), and (if `reg` is supplied) total OOS days the expert was
    actually used at prediction time.
    """
    rows = []
    for regime, imp_df in sorted(expert_imps.items()):
        n_folds = int(len(imp_df))
        # imp_df: index = fold dates, columns = features. Compute the mean
        # absolute importance across (fold × feature), a single scalar
        # indicating how active the expert was over its life.
        mean_abs_imp = float(imp_df.abs().mean().mean())
        oos_days = (
            int((reg["active_regime"].astype("string") == regime).sum())
            if reg is not None and not reg.empty else None
        )
        rows.append({
            "regime":              regime,
            "n_folds_trained":     n_folds,
            "mean_abs_importance": mean_abs_imp,
            "oos_days_used":       oos_days,
        })
    return pd.DataFrame(rows).sort_values("regime")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def _print_df(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> None:
    if df is None or df.empty:
        print("  (no data)")
        return
    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 50,
        "display.width", 200,
        "display.float_format", float_fmt.format,
    ):
        print(df.to_string())


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------


def _plot(run_dir: str, cmi_log: pd.DataFrame, reg: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

    # 1. CMI winner timeline as a categorical strip.
    cands = sorted(cmi_log["winner"].unique())
    palette = {c: f"C{i}" for i, c in enumerate(cands)}
    y = cmi_log["winner"].map({c: i for i, c in enumerate(cands)})
    axes[0].scatter(cmi_log.index, y,
                    c=[palette[w] for w in cmi_log["winner"]],
                    s=40, marker="s")
    axes[0].set_yticks(range(len(cands)))
    axes[0].set_yticklabels(cands)
    axes[0].set_title("CMI tournament winner per fold")
    axes[0].grid(alpha=0.3)

    # 2. Per-OOS-date active regime, coloured by separator winner.
    if not reg.empty:
        for w, sub in reg.groupby("winner_separator"):
            axes[1].scatter(sub.index, sub["active_regime"],
                            label=w, alpha=0.7, s=8)
        axes[1].set_title("Active regime per OOS date (coloured by separator)")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(alpha=0.3)

    out = os.path.join(run_dir, "ebm_homoe_usage_plot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze HO-MoE separator / regime / expert usage from the "
            "artifacts written by train_ebm_signal.py --ho_moe."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run_id", default=None,
                    help="Run ID under ./reports/strategies/")
    ap.add_argument("--run_dir", default=None,
                    help="Full path to the run directory (overrides --run_id)")
    ap.add_argument("--plot", action="store_true",
                    help="Also write ebm_homoe_usage_plot.png")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")

    cmi_log = _load_cmi_log(run_dir)
    reg = _load_regime_timeline(run_dir)
    expert_imps = _load_expert_importances(run_dir)

    if cmi_log is None and reg is None and not expert_imps:
        raise SystemExit(
            f"No HO-MoE artifacts found in {run_dir}. Was this run trained "
            "with --ho_moe?")

    print(f"Run directory : {run_dir}")
    if cmi_log is not None:
        print(f"  CMI log     : {len(cmi_log)} folds  "
              f"({cmi_log.index.min().date()} → {cmi_log.index.max().date()})")
    if reg is not None:
        print(f"  Regime time : {len(reg)} OOS days  "
              f"({reg.index.min().date()} → {reg.index.max().date()})")
    if expert_imps:
        print(f"  Experts     : {sorted(expert_imps.keys())}")

    consolidated_rows: list[dict] = []

    # 1. Winner distribution (overall) ------------------------------------
    if cmi_log is not None:
        _print_section("Separator winner — overall")
        winner_summary = summarize_winner(cmi_log)
        _print_df(winner_summary)
        for _, r in winner_summary.iterrows():
            consolidated_rows.append({
                "kind":  "winner",
                "key":   r["candidate"],
                "value": r["n_folds_won"],
                "pct":   r["win_share"],
                "extra": f"mean_raw_cmi={r['mean_raw_cmi']:.4g}  "
                         f"mean_ema_cmi={r['mean_ema_cmi']:.4g}",
            })

        _print_section("Separator winner — by calendar quarter")
        _print_df(summarize_winner_by_quarter(cmi_log), float_fmt="{:.0f}")

        _print_section("Separator winner — consecutive-fold runs")
        _print_df(summarize_winner_streaks(cmi_log), float_fmt="{:.0f}")

    # 2. Regime usage at OOS time -----------------------------------------
    if reg is not None and not reg.empty:
        _print_section(
            "Regime usage at OOS prediction time "
            "(by (separator winner, active regime))")
        regime_usage = summarize_regime_usage(reg)
        _print_df(regime_usage)
        for _, r in regime_usage.iterrows():
            consolidated_rows.append({
                "kind":  "regime",
                "key":   f"{r['active_regime']}|{r['winner_separator']}",
                "value": int(r["n_days"]),
                "pct":   float(r["share_overall"]),
                "extra": (
                    f"within_winner_share={r['share_within_winner']:.3f}  "
                    f"expert_available_pct={r['expert_available_pct']:.3f}"
                ),
            })

        # Regime distribution irrespective of separator.
        _print_section("Regime usage at OOS — collapsed over separators")
        collapsed = (
            reg.groupby("active_regime", dropna=False)
            .size().rename("n_days").to_frame()
        )
        collapsed["share"] = collapsed["n_days"] / collapsed["n_days"].sum()
        _print_df(collapsed)

    # 3. Expert coverage --------------------------------------------------
    if expert_imps:
        _print_section("Expert coverage")
        exp_summary = summarize_expert_coverage(expert_imps, reg)
        _print_df(exp_summary)
        for _, r in exp_summary.iterrows():
            consolidated_rows.append({
                "kind":  "expert_coverage",
                "key":   r["regime"],
                "value": int(r["n_folds_trained"]),
                "pct":   np.nan,
                "extra": (
                    f"mean_abs_importance={r['mean_abs_importance']:.4f}  "
                    f"oos_days_used={r['oos_days_used']}"
                ),
            })

    # 4. Save consolidated CSV --------------------------------------------
    if consolidated_rows:
        consolidated = pd.DataFrame(consolidated_rows)
        out_csv = os.path.join(run_dir, "ebm_homoe_usage_summary.csv")
        consolidated.to_csv(out_csv, index=False)
        print(f"\nConsolidated summary saved → {out_csv}")

    if args.plot and cmi_log is not None and reg is not None:
        _plot(run_dir, cmi_log, reg)


if __name__ == "__main__":
    main()
