"""M2: technical indicators. All use only past data (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import TechnicalIndicator


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, range 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss != 0, 100.0).where(gain.notna())


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_hist": macd - sig})


def calc_bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + n_std * sd, "bb_lower": mid - n_std * sd})


def calc_sma_ema(close: pd.Series) -> pd.DataFrame:
    out = {f"sma_{n}": close.rolling(n).mean() for n in (7, 21, 50)}
    out.update({f"ema_{n}": close.ewm(span=n, adjust=False).mean() for n in (9, 21)})
    return pd.DataFrame(out)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI(14), MACD(12,26,9), BB(20), SMA(7,21,50), EMA(9,21), volume change %."""
    out = df.copy()
    c = out["close"]
    out["rsi"] = calc_rsi(c)
    out = pd.concat([out, calc_macd(c), calc_bollinger(c), calc_sma_ema(c)], axis=1)
    out["volume_change"] = out["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return out


def store_indicators(session: Session, ticker: str, df: pd.DataFrame) -> int:
    """Save the indicator columns kept in the technical_indicators table."""
    have = set(session.scalars(select(TechnicalIndicator.date).where(TechnicalIndicator.ticker == ticker)))
    f = lambda v: None if pd.isna(v) else float(v)
    new = 0
    for r in df.itertuples(index=False):
        d = pd.Timestamp(r.date).date()
        if d in have or pd.isna(r.rsi):
            continue
        session.add(TechnicalIndicator(ticker=ticker, date=d, rsi=f(r.rsi), macd=f(r.macd),
                                       macd_signal=f(r.macd_signal), bb_upper=f(r.bb_upper),
                                       bb_lower=f(r.bb_lower), sma_21=f(r.sma_21), ema_9=f(r.ema_9)))
        new += 1
    session.commit()
    return new
