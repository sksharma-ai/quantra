"""M2: feature engineering + target for next-day direction.

Features are scale-free (returns, ratios) so the model does not learn raw price levels.
Row t only uses data up to day t. Target for row t is 1 if close[t+1] > close[t].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.indicators import add_indicators

N_LAGS = 20

BASE_FEATURES = [
    "rsi", "macd", "macd_signal", "macd_hist", "bb_position", "bb_width",
    "close_to_sma7", "close_to_sma21", "close_to_sma50", "close_to_ema9", "close_to_ema21",
    "volume_change", "daily_return", "overnight_gap", "std_7", "std_30",
]
LAG_FEATURES = [f"ret_lag_{k}" for k in range(1, N_LAGS + 1)]
FEATURE_COLUMNS = BASE_FEATURES + LAG_FEATURES


def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(prices)
    c = df["close"]
    df["daily_return"] = c.pct_change()
    df["overnight_gap"] = df["open"] / c.shift(1) - 1
    df["std_7"] = df["daily_return"].rolling(7).std()
    df["std_30"] = df["daily_return"].rolling(30).std()
    band = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_position"] = (c - df["bb_lower"]) / band
    df["bb_width"] = band / df["bb_mid"]
    for n in (7, 21, 50):
        df[f"close_to_sma{n}"] = c / df[f"sma_{n}"] - 1
    for n in (9, 21):
        df[f"close_to_ema{n}"] = c / df[f"ema_{n}"] - 1
    lags = {f"ret_lag_{k}": df["daily_return"].shift(k) for k in range(1, N_LAGS + 1)}
    df = pd.concat([df, pd.DataFrame(lags)], axis=1)
    nxt = c.shift(-1)
    df["target"] = (nxt > c).astype(float).where(nxt.notna())  # last row stays NaN
    return df


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Rows with a complete feature set. Latest row keeps target=NaN (used for live prediction)."""
    df = engineer_features(prices)
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
