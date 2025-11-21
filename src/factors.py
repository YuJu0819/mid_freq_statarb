# src/factors.py
import pandas as pd
import numpy as np

# --- ADX Calculation (Manual) ---


def calculate_adx(df: pd.DataFrame, length: int = 30):
    """
    Calculates the Average Directional Index (ADX) manually.
    Expects DataFrame with 'high', 'low', 'futures_close' columns.
    """
    df_adx = df.copy()
    alpha = 1 / length

    # Ensure columns exist, use defaults if not (for robustness)
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

# --- Regime / Environment Factors ---


def calc_btc_regimes(btc_df: pd.DataFrame):
    """Calculates market-wide regimes (vol & trend) using BTC as proxy."""
    btc_df = btc_df.copy()
    btc_df['returns'] = btc_df['futures_close'].pct_change()

    # Volatility Regime
    btc_df['volatility'] = btc_df['returns'].rolling(window=90).std()
    vol_low_q = btc_df['volatility'].quantile(0.25)
    vol_high_q = btc_df['volatility'].quantile(0.75)
    btc_df['volatility_regime'] = 'Medium Volatility'
    btc_df.loc[btc_df['volatility'] < vol_low_q,
               'volatility_regime'] = 'Low Volatility'
    btc_df.loc[btc_df['volatility'] > vol_high_q,
               'volatility_regime'] = 'High Volatility'

    # Trend Regime
    btc_df['adx'] = calculate_adx(btc_df, length=14)
    btc_df['trend_regime'] = 'Weak Trend'
    btc_df.loc[btc_df['adx'] > 25, 'trend_regime'] = 'Strong Trend'
    btc_df.loc[btc_df['adx'] < 20, 'trend_regime'] = 'Ranging'

    return btc_df[['ts', 'volatility_regime', 'trend_regime', 'adx']]

# --- Per-Asset Factors (Updated for Vectorization) ---


def calc_price_mom(prices, lookback: int, smooth_lookback: int):
    """Smoothed Price Rate of Change. Accepts Series or DataFrame."""
    return prices.pct_change(lookback).rolling(smooth_lookback).mean()


def calc_oi_mom(open_interest, lookback: int, smooth_lookback: int):
    """Smoothed Open Interest Rate of Change. Accepts Series or DataFrame."""
    return open_interest.pct_change(lookback).rolling(smooth_lookback).mean()


def calc_basis_mom(basis, prices, lookback: int, smooth_lookback: int):
    """Smoothed Basis Momentum. Accepts Series or DataFrame."""
    # Avoid division by zero
    safe_prices = prices.replace(0, 1e-12)
    basis_norm = basis / safe_prices
    return basis_norm.diff(lookback).rolling(smooth_lookback).mean()


def calc_vol_ratio_signal(volume_ratio, rolling_lookback: int, diff_lookback: int):
    """2^x signal based on diff of rolling avg volume ratio. Accepts Series or DataFrame."""
    vol_ratio_diff = volume_ratio.rolling(
        rolling_lookback).mean().diff(diff_lookback)

    # 2^x, default to 1 (2^0) if diff is NaN
    return 2 ** vol_ratio_diff.fillna(0.0)


def calc_funding_zscore(funding_rates, lookback: int):
    """Time-series Z-score of funding rate. Accepts Series or DataFrame."""
    mean = funding_rates.rolling(lookback).mean()
    std = funding_rates.rolling(lookback).std().replace(0, 1e-12)
    return (funding_rates - mean) / std


def calc_volatility(prices, lookback: int):
    """Rolling return volatility. Accepts Series or DataFrame."""
    return prices.pct_change().rolling(lookback).std()


def calc_skewness(prices, lookback: int):
    """Rolling skewness of daily returns. Accepts Series or DataFrame."""
    returns = prices.pct_change()
    return returns.rolling(window=lookback, min_periods=30).skew()
