"""
Score Inspector — diagnose universe health and factor activity per strategy.

Usage:
    python src/debug/score_inspector.py --strategy momentum --factor trend_score
    python src/debug/score_inspector.py --strategy reversal  --factor oi_z_score
    python src/debug/score_inspector.py --strategy combo     --run_id batch_2024_v1 --factor composite_alpha
    python src/debug/score_inspector.py --strategy momentum  --all_factors

Available factors per strategy:
    momentum : trend_score, sentiment_score, funding_z_score, volatility,
               price_roc, final_score
    reversal : oi_z_score, liquidation_shock, regime_score,
               interaction_alpha, final_signal
    combo    : composite_alpha
"""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_CONFIG = {
    "momentum": {
        "csv": "reports/score_inspection.csv",
        "save_dir": "reports",
        "factors": ["trend_score", "sentiment_score", "funding_z_score",
                    "volatility", "price_roc", "final_score"],
        "default_factor": "trend_score",
    },
    "reversal": {
        "csv": "reports_reversal_daily/score_inspection.csv",
        "save_dir": "reports_reversal_daily",
        "factors": ["oi_z_score", "liquidation_shock", "regime_score",
                    "interaction_alpha", "final_signal"],
        "default_factor": "oi_z_score",
    },
    "combo": {
        "csv": None,        # resolved from --run_id at runtime
        "save_dir": None,
        "factors": ["composite_alpha"],
        "default_factor": "composite_alpha",
    },
}


def analyze_active_universe(file_path: str, score_col: str, save_dir: str):
    """
    For a given factor column, counts per day how many symbols have a
    valid (non-zero, non-NaN) value, revealing the true active universe size.
    Saves a PNG to save_dir.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    if score_col not in df.columns:
        print(f"Column '{score_col}' not found. Available: {list(df.columns)}")
        return

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])

    print(f"Analyzing '{score_col}'...")

    active_count  = df.groupby("ts")[score_col].apply(
        lambda x: ((x != 0) & x.notna()).sum()
    )
    total_records = df.groupby("ts")[score_col].count()
    active_ratio  = (active_count / total_records * 100).fillna(0)

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(active_count.index, active_count,
             label=f"Active ({score_col} ≠ 0)", color="steelblue", linewidth=1.5)
    ax1.plot(total_records.index, total_records,
             label="Total records (incl. 0)", color="gray",
             linestyle="--", alpha=0.5)
    ax1.set_title(f"Universe Health — Active Symbols: {score_col}")
    ax1.set_ylabel("Number of Assets")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(active_ratio.index, active_ratio.values,
                     alpha=0.6, color="seagreen")
    ax2.set_title(f"Active Ratio — % Universe with Non-Zero {score_col}")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    plt.xlabel("Date")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"universe_health_{score_col}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print(f"  Avg active/day : {active_count.mean():.1f}")
    print(f"  Min active/day : {int(active_count.min())}")
    print(f"  Max active/day : {int(active_count.max())}")
    print(f"  Avg active %   : {active_ratio.mean():.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Score inspector for all strategies")
    ap.add_argument("--strategy", default="momentum",
                    choices=["momentum", "reversal", "combo"])
    ap.add_argument("--factor", default=None,
                    help="Factor column to inspect (default: strategy-specific)")
    ap.add_argument("--run_id", default=None,
                    help="Run ID — required when --strategy combo")
    ap.add_argument("--all_factors", action="store_true",
                    help="Inspect every available factor for the strategy")
    args = ap.parse_args()

    cfg = STRATEGY_CONFIG[args.strategy]

    # Resolve paths for combo (run_id-dependent)
    if args.strategy == "combo":
        if not args.run_id:
            ap.error("--run_id is required for --strategy combo")
        base = f"./reports/strategies/{args.run_id}"
        csv_path = os.path.join(base, "score_inspection.csv")
        save_dir = base
    else:
        csv_path = cfg["csv"]
        save_dir = cfg["save_dir"]

    # Resolve factor list
    if args.all_factors:
        factors_to_run = cfg["factors"]
    else:
        factor = args.factor or cfg["default_factor"]
        if factor not in cfg["factors"]:
            print(f"Warning: '{factor}' not in known factors for "
                  f"{args.strategy} {cfg['factors']}. Proceeding anyway.")
        factors_to_run = [factor]

    for factor in factors_to_run:
        analyze_active_universe(csv_path, factor, save_dir)


if __name__ == "__main__":
    main()
