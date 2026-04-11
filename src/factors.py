# src/factors.py
import pandas as pd
import numpy as np
from typing import Dict


def calculate_adx(df: pd.DataFrame, length: int = 14):
    df_adx = df.copy()
    alpha = 1 / length
    high = df_adx.get('high', df_adx['futures_close'])
    low = df_adx.get('low', df_adx['futures_close'])
    close = df_adx['futures_close']
    df_adx['h-l'] = high - low
    df_adx['h-pc'] = abs(high - close.shift(1))
    df_adx['l-pc'] = abs(low - close.shift(1))
    df_adx['tr'] = df_adx[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df_adx['dm_plus'] = (high - high.shift(1))
    df_adx['dm_minus'] = (low.shift(1) - low)
    df_adx['dm_plus'] = df_adx['dm_plus'].where(
        (df_adx['dm_plus'] > df_adx['dm_minus']) & (df_adx['dm_plus'] > 0), 0)
    df_adx['dm_minus'] = df_adx['dm_minus'].where(
        (df_adx['dm_minus'] > df_adx['dm_plus']) & (df_adx['dm_minus'] > 0), 0)
    df_adx['atr'] = df_adx['tr'].ewm(alpha=alpha, adjust=False).mean()
    df_adx['dm_plus_smoothed'] = df_adx['dm_plus'].ewm(
        alpha=alpha, adjust=False).mean()
    df_adx['dm_minus_smoothed'] = df_adx['dm_minus'].ewm(
        alpha=alpha, adjust=False).mean()
    di_plus = 100 * (df_adx['dm_plus_smoothed'] /
                     df_adx['atr'].replace(0, 1e-12))
    di_minus = 100 * (df_adx['dm_minus_smoothed'] /
                      df_adx['atr'].replace(0, 1e-12))
    di_sum = (di_plus + di_minus).replace(0, 1e-12)
    dx = 100 * (abs(di_plus - di_minus) / di_sum)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx


def calc_market_regimes(market_data: Dict[str, pd.DataFrame]):
    """
    Calculates market-wide regimes (vol, trend, skew) using an equal-weighted 
    basket of assets (e.g., BTC, ETH, SOL) as the proxy.
    """
    returns_list = []
    adx_list = []

    for sym, df in market_data.items():
        if df.empty:
            continue
        ret = df['futures_close'].pct_change()
        ret.name = sym
        returns_list.append(ret)

        adx = calculate_adx(df, length=14)
        adx.name = sym
        adx_list.append(adx)

    if not returns_list:
        return pd.DataFrame()

    # 2. Create Aggregated Market Metrics
    returns_df = pd.concat(returns_list, axis=1)
    market_returns = returns_df.mean(axis=1)

    adx_df = pd.concat(adx_list, axis=1)
    market_adx = adx_df.mean(axis=1)

    # 3. Calculate Regime Indicators
    # FIX: Initialize with None for name to avoid conflict, then reset_index later
    regimes_df = pd.DataFrame(index=market_returns.index)

    # Volatility
    volatility = market_returns.rolling(window=30).std()
    vol_low_q = volatility.quantile(0.25)
    vol_high_q = volatility.quantile(0.75)

    regimes_df['volatility_regime'] = 'Medium Volatility'
    regimes_df.loc[volatility < vol_low_q,
                   'volatility_regime'] = 'Low Volatility'
    regimes_df.loc[volatility > vol_high_q,
                   'volatility_regime'] = 'High Volatility'

    # Trend
    regimes_df['adx'] = market_adx
    regimes_df['trend_regime'] = 'Weak Trend'
    regimes_df.loc[market_adx > 25, 'trend_regime'] = 'Strong Trend'
    regimes_df.loc[market_adx < 20, 'trend_regime'] = 'Ranging'

    # Skew
    skewness = market_returns.rolling(window=90).skew()
    regimes_df['skew_regime'] = 'Neutral Skew'
    regimes_df.loc[skewness < -0.5, 'skew_regime'] = 'Negative Skew'
    regimes_df.loc[skewness > 0.5, 'skew_regime'] = 'Positive Skew'

    # FIX: Cleanly convert index to column to remove ambiguity
    regimes_df = regimes_df.reset_index()

    # Ensure the time column is named 'ts'
    # If the index was unnamed, reset_index creates 'index'. If named 'ts', it creates 'ts'.
    if 'ts' not in regimes_df.columns:
        # Assuming the first column is the time index
        regimes_df.rename(columns={regimes_df.columns[0]: 'ts'}, inplace=True)

    return regimes_df[['ts', 'volatility_regime', 'trend_regime', 'skew_regime', 'adx']]

# --- Per-Asset Factors (Unchanged) ---


def calc_price_mom(prices, lookback: int, smooth_lookback: int):
    return prices.pct_change(lookback).rolling(smooth_lookback).mean()


def calc_oi_mom(open_interest, lookback: int, smooth_lookback: int):
    return open_interest.pct_change(lookback).rolling(smooth_lookback).mean()


def calc_basis_mom(basis, prices, lookback: int, smooth_lookback: int):
    # Fix: Fill NaNs in basis with 0.0 to prevent one missing spot day from killing momentum
    clean_basis = basis.fillna(0.0)
    safe_prices = prices.replace(0, 1e-12)
    basis_norm = clean_basis / safe_prices
    return basis_norm.diff(lookback).rolling(smooth_lookback, min_periods=1).mean().fillna(0.0)


def calc_vol_ratio_signal(volume_ratio, rolling_lookback: int, diff_lookback: int):
    vol_ratio_diff = volume_ratio.rolling(
        rolling_lookback).mean().diff(diff_lookback)
    return 2 ** vol_ratio_diff.fillna(0.0)


def calc_funding_zscore(funding_rates, lookback: int):
    mean = funding_rates.rolling(lookback, min_periods=20).mean()
    std = funding_rates.rolling(
        lookback, min_periods=20).std().replace(0, 1e-12)
    return (funding_rates - mean) / std


def calc_cs_zscore(target):
    target = target.replace(0, np.nan)
    mean = np.mean(target, axis=0)
    std = np.std(target, axis=0)
    return (target - mean) / std


def calc_beta_df(closes_wide, lookback):
    returns_wide = closes_wide.pct_change()  # Do not fillna(0) yet

    # Create Proxy (BTC/ETH/SOL)
    # Filter columns that exist in your closes_wide
    proxy_assets = [c for c in ['BTCUSDT', 'ETHUSDT',
                                'SOLUSDT'] if c in closes_wide.columns]

    if not proxy_assets:
        market_returns = returns_wide.mean(axis=1)  # Fallback
    else:
        market_returns = returns_wide[proxy_assets].mean(axis=1)

    # Calculate Rolling Beta
    market_var = market_returns.rolling(window=lookback).var()
    rolling_cov = returns_wide.rolling(window=lookback).cov(market_returns)

    beta_df = rolling_cov.div(market_var, axis=0)
    # Calculate the mean across the cross-section for each timestamp
    cs_mean_beta = beta_df.mean(axis=1)

    # Fill NaNs in each row with the mean of that row
    beta_df = beta_df.apply(lambda row: row.fillna(
        cs_mean_beta[row.name]), axis=1)
    # Fill missing betas with 1.0 (Assume correlation if unknown)
    return beta_df.clip(-3, 3)


def calc_volatility(prices, lookback: int):
    return prices.pct_change().rolling(lookback).std()


def calc_skewness(prices, lookback: int):
    returns = prices.pct_change()
    return returns.rolling(window=lookback, min_periods=30).skew()
