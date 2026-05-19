"""
Temporal-bag matrix construction and EBM ensemble fitting, factored out
of src/scripts/train_ebm_signal.py during the phase-4b refactor.

These two functions were previously nested inside `walk_forward`, closing
over `block_size`, `target_horizon`, `rng`, `use_block_bagging`,
`n_outer_bags`, `bag_symbol_frac`, and `bag_sym_excluded_as_val`. They are
promoted here to module-level with all closure variables turned into
explicit keyword-only parameters. The training script keeps thin local
closures with the original signatures so every interior call site is
unchanged.

The HO-MoE trainer (`walk_forward_ho_moe`) has its own simpler
`_make_temporal_bags` / `_train_ebm_ensemble` pair without symbol
subsampling — that pair is structurally different (same names, different
bodies) and stays inside the HO-MoE function for now.
"""
from __future__ import annotations

import numpy as np

from interpret.glassbox import ExplainableBoostingRegressor

from .ebm_utils import _block_bootstrap_counts


def make_temporal_bags(
    n: int,
    n_bags: int,
    use_blocks: bool,
    date_arr: "np.ndarray | None" = None,
    symbol_arr: "np.ndarray | None" = None,
    symbol_frac: float = 1.0,
    sym_excluded_as_val: bool = False,
    *,
    block_size: int,
    target_horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a (n_samples × n_bags) bags matrix with temporal validation.

    Returns an int16 matrix where each entry is:
      k > 0 : row is in this bag's training set, sampled k times
              (block-bootstrap-with-replacement count for the row's date,
              zero-ed if the row's symbol is not in the bag's symbol subset)
      -1    : row is in this bag's validation holdout (last val_dates,
              identical across bags for early-stopping consistency).
              When `sym_excluded_as_val=True` the symbols this bag
              dropped via subsampling ALSO join the validation set
              (per-bag, so each bag has a different val population) —
              blends temporal and cross-sectional generalization into
              the early-stopping signal. Default False keeps the
              legacy "pure temporal val" semantics.
       0    : row is excluded from this bag

    Diversity sources (all leak-free — sampling never touches the
    validation tail or any post-pred-date data):

      1. TIME diversity — block bootstrap WITH duplicates: each bag
         samples ~train_dates / block_size blocks of consecutive dates,
         with replacement. Counts > 1 are real (the canonical bootstrap
         effect), giving each bag ~63% unique training dates.
      2. CROSS-SECTIONAL diversity — when symbol_frac < 1.0, each bag
         keeps a random subset of symbols (without replacement). Other
         symbols are zeroed out for that bag. Bag-to-bag the symbol
         subsets differ, so each bag's CS panel is decorrelated.

    Validation rows are NEVER touched by either sampler — they remain
    -1 across all bags, and date sampling is restricted to the
    [0, train_date_cutoff) range.
    """
    if use_blocks and date_arr is not None:
        unique_dates = np.unique(date_arr)
        n_dates = len(unique_dates)
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        row_date_idx = np.array([date_to_idx[d] for d in date_arr])

        # Validation: last 20% of training dates (early-stopping holdout)
        val_dates = int(n_dates * 0.1)
        train_date_cutoff = n_dates - val_dates
        train_row_mask = row_date_idx < train_date_cutoff

        # Per-bag block count: aim for ~1× coverage of training dates,
        # producing canonical bootstrap behaviour (~63% unique).
        n_blocks_per_bag = max(
            1, int(np.ceil(train_date_cutoff / block_size)))

        # CS subsample setup
        do_sym_sub = (symbol_arr is not None) and (symbol_frac < 1.0)
        if do_sym_sub:
            unique_syms = np.unique(symbol_arr)
            n_syms = len(unique_syms)
            sym_to_idx = {s: i for i, s in enumerate(unique_syms)}
            row_sym_idx = np.array([sym_to_idx[s] for s in symbol_arr])
            keep_size = max(1, int(n_syms * symbol_frac))

        mat = np.zeros((n, n_bags), dtype=np.int16)
        mat[~train_row_mask, :] = -1  # validation rows pinned across bags

        for b in range(n_bags):
            # 1) TIME: sample blocks with replacement → per-date counts.
            #    Strictly within [0, train_date_cutoff) — no future leak.
            #    Pad to full date length so indexing by row_date_idx is
            #    safe for validation rows (their counts are zero anyway).
            tr_counts = _block_bootstrap_counts(
                train_date_cutoff, block_size, n_blocks_per_bag, rng)
            date_counts = np.zeros(n_dates, dtype=np.int16)
            date_counts[:train_date_cutoff] = tr_counts
            row_counts = date_counts[row_date_idx]  # safe broadcast

            # 2) CS: optional per-bag symbol subsample
            if do_sym_sub:
                keep_idx = rng.choice(
                    n_syms, size=keep_size, replace=False)
                sym_keep_mask = np.zeros(n_syms, dtype=bool)
                sym_keep_mask[keep_idx] = True
                sym_active = sym_keep_mask[row_sym_idx]
            else:
                sym_active = np.ones(n, dtype=bool)

            # 3) Combine: training rows that are in selected dates AND
            #    selected symbols receive their date count; others 0.
            bag_col = np.where(
                train_row_mask & sym_active & (row_counts > 0),
                row_counts, 0,
            ).astype(np.int16)
            # Opt-in: symbol-excluded training rows join this bag's
            # validation set instead of being skipped. Done BEFORE the
            # time-based pin so the time-based -1 takes precedence on
            # the validation tail (where sym subsampling is irrelevant).
            if sym_excluded_as_val and do_sym_sub:
                sym_excluded_train = (
                    train_row_mask & ~sym_active)
                bag_col[sym_excluded_train] = -1
            # Restore validation pin for this column (always -1)
            bag_col[~train_row_mask] = -1
            mat[:, b] = bag_col
        return mat
    else:

        if date_arr is not None:
            unique_dates = np.unique(date_arr)
            n_dates = len(unique_dates)
            date_to_idx = {d: i for i, d in enumerate(unique_dates)}
            row_date_idx = np.array([date_to_idx[d] for d in date_arr])
            # val_dates = min(max(target_horizon, block_size),
            #                 int(n_dates * 0.2))
            val_dates = int(n_dates * 0.2)
            train_date_cutoff = n_dates - val_dates
            train_row_mask = row_date_idx < train_date_cutoff
            # breakpoint()
        else:
            val_size = min(max(target_horizon, block_size), n // 5)
            train_row_mask = np.ones(n, dtype=bool)
            train_row_mask[n - val_size:] = False
            raise ValueError(
                "_make_temporal_bags requires date_arr to compute a "
                "temporally correct validation split. Passing date_arr=None "
                "risks look-forward bias if rows are not sorted by date."
            )

        mat = np.zeros((n, n_bags), dtype=np.int8)
        mat[train_row_mask, :] = 1
        mat[~train_row_mask, :] = -1
        return mat


def train_ebm_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    kwargs: dict,
    date_arr: "np.ndarray | None" = None,
    symbol_arr: "np.ndarray | None" = None,
    force_no_bagging: bool = False,
    *,
    use_block_bagging: bool,
    n_outer_bags: int,
    block_size: int,
    bag_symbol_frac: float,
    bag_sym_excluded_as_val: bool,
    target_horizon: int,
    rng: np.random.Generator,
) -> list:
    """Train a single or block-bagged EBM ensemble. Returns list of models.

    Block-bagging uses a manual bags matrix with two leak-free diversity
    sources: (1) block bootstrap WITH replacement on training-window dates
    (per-date counts, ~63% unique → real variance reduction), and (2)
    per-bag symbol subsampling when bag_symbol_frac < 1.0.

    Non-block-bagging (or force_no_bagging=True) fits a plain EBM on the
    full training set.  Expert models pass force_no_bagging=True because
    their per-regime subsets are already small and external bagging would
    shrink effective training history too aggressively.
    """
    n = len(X)
    if use_block_bagging and not force_no_bagging:
        bags_matrix = make_temporal_bags(
            n, n_outer_bags, use_blocks=True,
            date_arr=date_arr, symbol_arr=symbol_arr,
            symbol_frac=bag_symbol_frac,
            sym_excluded_as_val=bag_sym_excluded_as_val,
            block_size=block_size,
            target_horizon=target_horizon,
            rng=rng,
        )
        kwargs_bag = {**kwargs, "outer_bags": n_outer_bags}
        m = ExplainableBoostingRegressor(**kwargs_bag)
        m.fit(X, y, bags=bags_matrix)
        return [m]
    else:
        n_bags = kwargs.get("outer_bags", 8)
        bags_matrix = make_temporal_bags(
            n, n_bags, use_blocks=False, date_arr=date_arr,
            block_size=block_size,
            target_horizon=target_horizon,
            rng=rng,
        )
        kwargs_fixed = {**kwargs, "outer_bags": n_bags}
        m = ExplainableBoostingRegressor(**kwargs_fixed)
        m.fit(X, y, bags=bags_matrix)
        return [m]
