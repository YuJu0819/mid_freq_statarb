#!/bin/bash
# =============================================================================
# Full Backtest Pipeline
#
# Usage:
#   ./run_pipeline.sh                          # run all steps
#   ./run_pipeline.sh --skip-step1             # skip download_metrics
#   ./run_pipeline.sh --start-from 3           # start from step 3 onwards
#   ./run_pipeline.sh --only 4                 # run only step 4
#
# Steps:
#   1  download_metrics      (OI + ls_ratio archives — run once or monthly)
#   2  download_ls_ratio     (recent l/s ratio — run every <=25 days)
#   3  prepare_universe      (validate symbols + pre-cache prices)
#   4  backtest_momentum     (momentum strategy backtest)
#   5  backtest_reversal     (liquidation reversal backtest)
#   6  backtest_combo        (combined portfolio)
#   7  analyze_weights       (weight distribution, coverage, overlap analysis)
#   8  build_factor_panel    (flat ML factor panel for ML signal research)
#   9  train_ebm_signal      (walk-forward EBM signal generation)
#  10  backtest_ebm          (backtest EBM signal + full 3-strategy combo)
#  11  analyze_ebm_predictions  (inspect raw EBM scores + IC diagnostics)
# =============================================================================

set -e   # exit immediately on any error

# --- Configuration -----------------------------------------------------------
START_DATE="2024-01-01"
END_DATE="2025-12-31"
RUN_ID="batch_v2"
MOM_ALLOC="0.6"
REV_ALLOC="0.4"
LS_DAYS="30"          # days of l/s history to fetch in step 2
LS_INTERVAL="1d"      # interval for l/s ratio
MIN_COVERAGE="0.80"   # minimum price-data coverage to include a symbol
ROLLING=""            # set to "--rolling" to use rolling universe mode
REFRESH_UNIVERSE=0    # set to 1 via --refresh-universe to run universe refresh first
REFRESH_TOP_N="150"   # number of top symbols to use when refreshing
COVERAGE_THRESHOLD="0.85"  # overlap fraction below which a refresh is recommended
# -----------------------------------------------------------------------------

# --- Argument parsing --------------------------------------------------------
SKIP_STEPS=()
START_FROM=1
ONLY_STEP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-step*)
            # e.g. --skip-step1  --skip-step2
            N="${1//[^0-9]/}"
            SKIP_STEPS+=("$N")
            shift ;;
        --start-from)
            START_FROM="$2"; shift 2 ;;
        --only)
            ONLY_STEP="$2"; shift 2 ;;
        --start_date)
            START_DATE="$2"; shift 2 ;;
        --end_date)
            END_DATE="$2"; shift 2 ;;
        --run_id)
            RUN_ID="$2"; shift 2 ;;
        --rolling)
            ROLLING="--rolling"; shift ;;
        --refresh-universe)
            REFRESH_UNIVERSE=1; shift ;;
        --refresh-top-n)
            REFRESH_TOP_N="$2"; shift 2 ;;
        --coverage-threshold)
            COVERAGE_THRESHOLD="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Helper ------------------------------------------------------------------
should_run() {
    local STEP=$1
    # If --only is set, run only that step
    if [[ -n "$ONLY_STEP" ]]; then
        [[ "$STEP" == "$ONLY_STEP" ]] && return 0 || return 1
    fi
    # Skip if before START_FROM
    [[ "$STEP" -lt "$START_FROM" ]] && return 1
    # Skip if in SKIP_STEPS list
    for s in "${SKIP_STEPS[@]}"; do
        [[ "$STEP" == "$s" ]] && return 1
    done
    return 0
}

step_header() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  Step $1: $2"
    echo "──────────────────────────────────────────────────"
}

# --- Banner ------------------------------------------------------------------
echo "=========================================================="
echo "  Backtest Pipeline   run_id=$RUN_ID"
echo "  Period : $START_DATE → $END_DATE"
echo "  Steps  : 1=metrics  2=ls_ratio  3=universe  4=mom  5=rev  6=combo"
if [[ -n "$ROLLING" ]]; then
    echo "  Mode   : ROLLING UNIVERSE"
fi
if [[ "$REFRESH_UNIVERSE" -eq 1 ]]; then
    echo "  Refresh: universe refresh will run before step 1"
fi
if [[ ${#SKIP_STEPS[@]} -gt 0 ]]; then
    echo "  Skip   : ${SKIP_STEPS[*]}"
fi
if [[ "$START_FROM" -gt 1 ]]; then
    echo "  Start from step $START_FROM"
fi
if [[ -n "$ONLY_STEP" ]]; then
    echo "  Only step $ONLY_STEP"
fi
echo "=========================================================="

# --- Optional: Refresh universe (run before Step 1 when requested) ----------
if [[ "$REFRESH_UNIVERSE" -eq 1 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  Universe Refresh  (top-$REFRESH_TOP_N by 24h volume)"
    echo "──────────────────────────────────────────────────"
    echo "  Checking current coverage first ..."
    python -m src.scripts.data.refresh_universe \
        --check_coverage \
        --top_n "$REFRESH_TOP_N" \
        --coverage_threshold "$COVERAGE_THRESHOLD"

    echo ""
    echo "  Fetching new top-$REFRESH_TOP_N snapshot and updating config.yaml ..."
    python -m src.scripts.data.refresh_universe \
        --top_n "$REFRESH_TOP_N" \
        --apply \
        --download_data
fi

# --- Step 1: Download metrics (OI + ls_ratio archives) ----------------------
if should_run 1; then
    step_header 1 "download_metrics  (OI + ls_ratio archives)"
    python -m src.scripts.data.download_metrics
fi

# --- Step 2: Accumulate recent l/s ratio ------------------------------------
if should_run 2; then
    step_header 2 "download_ls_ratio  (last ${LS_DAYS} days)"
    python -m src.scripts.data.download_ls_ratio \
        --days "$LS_DAYS" \
        --interval "$LS_INTERVAL"
fi

# --- Step 3: Validate universe + pre-cache prices ---------------------------
if should_run 3; then
    step_header 3 "prepare_universe  ($START_DATE → $END_DATE)${ROLLING:+  [rolling]}"
    python -m src.scripts.data.prepare_universe \
        --start_date "$START_DATE" \
        --end_date   "$END_DATE" \
        --min_coverage "$MIN_COVERAGE" \
        $ROLLING
fi

# --- Step 4: Momentum backtest ----------------------------------------------
if should_run 4; then
    step_header 4 "backtest_momentum"
    python -m src.scripts.backtest.backtest_multi \
        --start_date "$START_DATE" \
        --end_date   "$END_DATE" \
        --run_id     "$RUN_ID"
fi

# --- Step 5: Reversal backtest ----------------------------------------------
if should_run 5; then
    step_header 5 "backtest_reversal"
    python -m src.scripts.backtest.backtest_reversal \
        --start_date "$START_DATE" \
        --end_date   "$END_DATE" \
        --run_id     "$RUN_ID"
fi

# --- Step 6: Combined portfolio ---------------------------------------------
if should_run 6; then
    step_header 6 "backtest_combo  (momentum + reversal)"
    python -m src.scripts.backtest.backtest_combo \
        --run_id     "$RUN_ID" \
        --start_date "$START_DATE" \
        --end_date   "$END_DATE" \
        --strategies momentum reversal \
        --method     mean_variance \
        --max_position 0.10
fi

# --- Step 7: Weight analysis -------------------------------------------------
if should_run 7; then
    step_header 7 "analyze_weights"
    python -m src.scripts.analysis.analyze_weights \
        --run_id "$RUN_ID"
fi

# --- Step 8: Build ML factor panel -------------------------------------------
if should_run 8; then
    step_header 8 "build_factor_panel"
    python -m src.scripts.build.build_factor_panel \
        --run_id     "$RUN_ID" \
        --start_date "$START_DATE" \
        --end_date   "$END_DATE"
fi

# --- Step 9: EBM signal ------------------------------------------------------
if should_run 9; then
    step_header 9 "train_ebm_signal"
    PANEL_PATH="./data/ml/factor_panel_${START_DATE}_${END_DATE}.parquet"
    python -m src.scripts.build.train_ebm_signal \
        --run_id        "$RUN_ID" \
        --panel_path    "$PANEL_PATH" \
        --target_col    ret_1d \
        --target_horizon 1 \
        --target_type   raw \
        --feature_norm  cs \
        --train_window  252 \
        --retrain_freq  21 \
        --min_train_periods 126 \
        --quantile      0.4 \
        --max_weight    0.10 \
        --max_rounds    200 \
        --interactions  10 \
        --features all \
        --weight_mode rank \
        --include_signals \
        --beta_neutral \
        --min_samples_leaf 60 \
        --n_outer_bags 3 \
        --block_size 63 \
        --embargo_pct 0.01 \
        --outer_bags 1 \
        --target_beta_neutral \
        --use_moe
fi

# --- Step 10: Backtest EBM signal --------------------------------------------
if should_run 10; then
    step_header 10 "backtest_ebm"

    # 10a. EBM signal alone
    echo "  [10a] EBM standalone backtest..."
    python -m src.scripts.backtest.backtest_combo \
        --run_id     "$RUN_ID" \
        --start_date "$START_DATE" \
        --end_date   "$END_DATE" \
        --strategies ebm \
        --method     linear \
        --max_position 0.10

    # 10b. Full 3-strategy combo (momentum + reversal + ebm)
    echo "  [10b] Full 3-strategy combo (momentum + reversal + ebm)..."
    python -m src.scripts.backtest.backtest_combo \
        --run_id     "$RUN_ID" \
        --start_date "$START_DATE" \
        --end_date   "$END_DATE" \
        --strategies momentum reversal ebm \
        --method     linear \
        --max_position 0.10

    # 10c. Weight analysis including EBM
    echo "  [10c] Weight analysis (all 3 strategies)..."
    python -m src.scripts.analysis.analyze_weights \
        --run_id     "$RUN_ID" \
        --strategies momentum reversal ebm
fi

# --- Step 11: Inspect raw EBM predictions (optional) ------------------------
if should_run 11; then
    step_header 11 "analyze_ebm_predictions"
    PANEL_PATH="./data/ml/factor_panel_${START_DATE}_${END_DATE}.parquet"
    python -m src.scripts.analysis.analyze_ebm_predictions \
        --run_id        "$RUN_ID" \
        --panel_path    "$PANEL_PATH" \
        --target_col    ret_1d \
        --target_horizon 1 \
        --neutralized
fi

echo ""
echo "=========================================================="
echo "  Pipeline complete.  Reports → ./reports/strategies/$RUN_ID"
echo "=========================================================="
