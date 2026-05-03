---
name: rolling_universe_system
description: Rolling universe refresh system added to address survivorship bias
type: project
---

# Rolling Universe System (added 2026-04-18)

**Why:** Static top-150 universe snapshot causes survivorship bias as delisted/demoted coins stay in the universe and new entrants are missed. For delisted coins, data is not accessible from Binance API but IS available via Binance Data Vision archives.

## New files:
- `src/data/rolling_universe.py` — `RollingUniverse` class: manages dated snapshots in `data/universe_snapshots/`, provides `get_symbols_for_date(date)`, `get_epochs(start, end)`, `save_epoch_universe()`, `get_validated_symbols_for_date()`
- `src/scripts/refresh_universe.py` — fetches fresh top-N from Binance Futures, diffs vs last snapshot, downloads Binance Vision metrics for new symbols, saves snapshot, optionally updates config.yaml

## Modified files:
- `src/scripts/prepare_universe.py` — added `--rolling` flag: validates per-epoch symbols using `RollingUniverse.get_epochs()`, handles mid-epoch listings by adjusting expected_days to actual data start, saves per-epoch YAML files + union universe YAML
- `run_pipeline.sh` — added `--rolling` flag (passes to step 3), `--refresh-universe` flag (runs refresh before step 1), `--refresh-top-n`, `--coverage-threshold`

## Data layout:
```
data/universe_snapshots/
├── snapshot_2026-04-18.yaml                                       # dated top-N snapshots
├── universe_2026-04-18_2026-10-17_snap_2026-04-18.yaml           # validated epoch universes
└── ...
```

## Recommended workflow:

**Semi-annual refresh:**
```bash
# Check coverage first
python -m src.scripts.refresh_universe --check_coverage

# Fetch new snapshot, download data for new symbols, update config.yaml
python -m src.scripts.refresh_universe --apply --download_data

# Run full pipeline with rolling universe
./run_pipeline.sh --rolling --start_date 2024-01-01 --end_date 2026-12-31

# Or one-shot automated:
./run_pipeline.sh --refresh-universe --rolling --start_date 2024-01-01 --end_date 2026-12-31
```

**How to apply:** When user asks about universe refresh, survivorship bias, or adding new coins, refer to this system. Backtest scripts (backtest_multi, backtest_reversal, etc.) still use the standard `load_validated_universe(start, end)` — the rolling mode also writes the union universe file so they work unchanged.
