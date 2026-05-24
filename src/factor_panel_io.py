"""
Factor panel I/O helpers.

Contents
--------
  load_metrics_store  read open_interest / ls_ratio per symbol from the
                      historical metrics archive (CSV) + the recent
                      ls_ratio parquet accumulation store
  attach_signals      left-join strategy weight parquets onto the panel by
                      (ts, symbol) so the EBM can train on (or alongside)
                      the strategy outputs

  load_panel          read a per-epoch factor-panel directory (current
                      default) or a legacy single-file .parquet path, and
                      return a PanelBundle that knows how to route a given
                      timestamp to the correct epoch's panel.
  PanelBundle         per-epoch panel container + lookup helpers.
  EpochEntry          one entry from the per-epoch manifest.
  is_panel_directory  layout sniffer.

Per-epoch layout (produced by build_factor_panel.py when rolling universe
is on):

  data/ml/factor_panel_<start>_<end>/
      manifest.yaml
      epoch_2024-01-01.parquet      # full [start, end] history,
      epoch_2024-07-01.parquet      # restricted to that epoch's universe
      ...

Each per-epoch parquet contains the FULL history but is restricted to that
epoch's universe of symbols; cross-sectional columns inside are computed
across only that universe. Walk-forward EBM training routes each fold to
the panel whose universe is active on the fold's prediction date.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from .data.loader import _load_local_metrics


def load_metrics_store(
    symbols: "list[str]",
    start_date: str,
    end_date: str,
    metrics_dir: str = "./data/metrics",
    ls_parquet_dir: str = "./data/ls_ratio",
) -> "tuple[dict[str, pd.Series], dict[str, pd.Series]]":
    """
    Load open_interest and ls_ratio for each symbol, merging two sources:

      Primary  : data/metrics/{symbol}_metrics_*.csv
                 Full historical archive (downloaded by download_metrics.py).
                 Contains both open_interest and ls_ratio.

      Fallback : data/ls_ratio/{symbol}_ls_ratio.parquet
                 Recent accumulation store (download_ls_ratio.py).
                 Covers only the last N days but has finer granularity.

    Returns
    -------
    oi_store  : {symbol: pd.Series(ts → open_interest)}
    ls_store  : {symbol: pd.Series(ts → ls_ratio)}

    Why this replaces the old load_ls_store + _load_local_oi pair:
      - _load_local_oi() used the wrong filename pattern
        ({symbol}-metrics-{date}.csv with hyphens) while the actual files
        use underscores ({symbol}_metrics_{date}.csv).
      - load_ls_store() read only the recent parquet accumulation store,
        missing all historical dates in the backtest window.
    """
    t0 = pd.to_datetime(start_date)
    t1 = pd.to_datetime(end_date)

    oi_store: "dict[str, pd.Series]" = {}
    ls_store: "dict[str, pd.Series]" = {}

    for sym in symbols:
        frames = []

        # ── Primary: historical metrics CSVs ─────────────────────────────────
        df_hist = _load_local_metrics(sym, metrics_dir)
        if not df_hist.empty:
            df_hist = df_hist[(df_hist["ts"] >= t0) & (df_hist["ts"] <= t1)]
            frames.append(df_hist)

        # ── Fallback: recent parquet accumulation store ───────────────────────
        parquet_path = os.path.join(ls_parquet_dir, f"{sym}_ls_ratio.parquet")
        if os.path.exists(parquet_path):
            try:
                df_rec = pd.read_parquet(parquet_path)
                df_rec["ts"] = pd.to_datetime(df_rec["ts"])
                df_rec = df_rec[(df_rec["ts"] >= t0) & (df_rec["ts"] <= t1)]
                frames.append(df_rec)
            except Exception as e:
                print(f"  [metrics_store] {sym} parquet: {e}")

        if not frames:
            continue

        merged = (
            pd.concat(frames)
            .sort_values("ts")
            .drop_duplicates("ts", keep="last")
            .reset_index(drop=True)
        )

        if "open_interest" in merged.columns:
            s = merged.set_index("ts")["open_interest"].dropna()
            if not s.empty:
                oi_store[sym] = s

        if "ls_ratio" in merged.columns:
            s = merged.set_index("ts")["ls_ratio"].dropna()
            if not s.empty:
                ls_store[sym] = s

    return oi_store, ls_store


def attach_signals(
    panel: pd.DataFrame,
    run_id: str,
    strategy_names: "list[str]" = ("momentum", "reversal"),
    signal_cols: "list[str]" = ("mom_signal", "rev_signal"),
) -> pd.DataFrame:
    """
    Loads weight parquets and left-joins them onto the panel by (ts, symbol).
    Missing entries become NaN (symbol not in that strategy's universe on that day).
    """
    base = f"./reports/strategies/{run_id}"
    for name, col in zip(strategy_names, signal_cols):
        path = os.path.join(base, f"{name}.parquet")
        if not os.path.exists(path):
            print(
                f"  [signals] {path} not found — column '{col}' will be NaN.")
            panel[col] = np.nan
            continue
        w = pd.read_parquet(path)
        if not pd.api.types.is_datetime64_any_dtype(w.index):
            w.index = pd.to_datetime(w.index)
        # Stack to long
        stacked = w.stack().reset_index()
        stacked.columns = ["ts", "symbol", col]
        panel = panel.merge(stacked, on=["ts", "symbol"], how="left")
        print(f"  [signals] Attached '{col}' from {path}")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch panel layout
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpochEntry:
    snapshot_date: str       # e.g. "2024-07-01"
    epoch_start:   str       # e.g. "2024-07-01"
    epoch_end:     str       # e.g. "2024-12-31"
    n_symbols:     int
    file:          str       # filename relative to the panel directory


@dataclass
class PanelBundle:
    """
    Result of `load_panel(path)`.

    Two shapes:
      - `single=True`  → legacy single-file panel; `panel` holds the DataFrame.
      - `single=False` → per-epoch directory; `epochs` lists the manifest
        entries and `panels` maps snapshot_date → DataFrame (eager by default,
        lazy when `load_panel(..., eager=False)`).

    Use `get_panel_for_date(ts)` to fetch the right per-epoch DataFrame for
    a given prediction date — the universe whose epoch covers `ts`.
    """
    single: bool
    root: str
    panel: "pd.DataFrame | None" = None
    epochs: "list[EpochEntry]" = field(default_factory=list)
    panels: "dict[str, pd.DataFrame]" = field(default_factory=dict)

    def epoch_for_date(self, ts) -> "EpochEntry | None":
        if self.single or not self.epochs:
            return None
        t = pd.Timestamp(ts)
        for ep in self.epochs:
            es = pd.Timestamp(ep.epoch_start)
            ee = (pd.Timestamp(ep.epoch_end)
                  + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            if es <= t <= ee:
                return ep
        return None

    def get_panel_for_date(self, ts) -> "pd.DataFrame | None":
        if self.single:
            return self.panel
        ep = self.epoch_for_date(ts)
        if ep is None:
            return None
        if ep.snapshot_date not in self.panels:
            self.panels[ep.snapshot_date] = pd.read_parquet(
                os.path.join(self.root, ep.file))
        return self.panels[ep.snapshot_date]

    def get_panel_by_snap(self, snapshot_date: str) -> "pd.DataFrame | None":
        if self.single:
            return self.panel
        if snapshot_date in self.panels:
            return self.panels[snapshot_date]
        for ep in self.epochs:
            if ep.snapshot_date == snapshot_date:
                self.panels[snapshot_date] = pd.read_parquet(
                    os.path.join(self.root, ep.file))
                return self.panels[snapshot_date]
        return None

    def iter_panels(self):
        """Yield (snapshot_date, DataFrame) pairs in epoch order."""
        if self.single:
            yield "single", self.panel
            return
        for ep in self.epochs:
            yield ep.snapshot_date, self.get_panel_by_snap(ep.snapshot_date)


def is_panel_directory(path: str) -> bool:
    """True if `path` is a per-epoch panel directory (contains manifest.yaml)."""
    return (os.path.isdir(path)
            and os.path.exists(os.path.join(path, "manifest.yaml")))


def load_panel(path: str, eager: bool = True) -> PanelBundle:
    """
    Load a factor panel from either a single-file (.parquet) path or a
    per-epoch directory.

    Parameters
    ----------
    path  : either a .parquet file (legacy single-universe panel) or a
            directory produced by build_factor_panel.py in per-epoch mode.
    eager : when True (default), all per-epoch parquets are read into RAM at
            load time (≈80 MB per epoch × ~9 epochs ≈ 700 MB). Set False to
            lazy-load each epoch's panel on first access.

    Returns
    -------
    PanelBundle (see class docstring).
    """
    if path.endswith(".parquet") and os.path.isfile(path):
        return PanelBundle(
            single=True, root=os.path.dirname(path),
            panel=pd.read_parquet(path),
        )

    if is_panel_directory(path):
        with open(os.path.join(path, "manifest.yaml")) as f:
            manifest = yaml.safe_load(f)
        epochs = [EpochEntry(**ep) for ep in manifest["epochs"]]
        panels: dict[str, pd.DataFrame] = {}
        if eager:
            for ep in epochs:
                panels[ep.snapshot_date] = pd.read_parquet(
                    os.path.join(path, ep.file))
        return PanelBundle(
            single=False, root=path, epochs=epochs, panels=panels,
        )

    raise FileNotFoundError(
        f"Panel path not found: {path}. Expected either a .parquet file or "
        f"a directory containing manifest.yaml.")
