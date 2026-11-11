"""M1: price data ingestion (yfinance) with CSV cache fallback and DB storage."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from db.models import PriceData, Stock

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
COLS = ["date", "open", "high", "low", "close", "volume", "adj_close"]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('^', '_')}_prices.csv"


def _download(ticker: str, period: str, start: str | None, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so tests run without network libs

    kw = dict(auto_adjust=False, progress=False)
    if start:
        return yf.download(ticker, start=start, end=end, **kw)
    return yf.download(ticker, period=period, **kw)


def normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw yfinance frame -> clean frame matching the price_data table rules."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance versions
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df = df[COLS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)]
    # DB rule: high >= open, low <= open. Vendor rounding can break it, so clamp.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["volume"] = df["volume"].astype("int64")
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_price_data(ticker: str, period: str = "2y", start: str | None = None,
                     end: str | None = None) -> pd.DataFrame:
    """Download OHLCV. On any failure, fall back to the last cached copy (NFR-04)."""
    try:
        df = normalize_prices(_download(ticker, period, start, end))
        if df.empty:
            raise ValueError(f"no data returned for {ticker}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_cache_path(ticker), index=False)
        return df
    except Exception as exc:  # network down, rate limit, bad ticker...
        path = _cache_path(ticker)
        if path.exists():
            log.warning("fetch failed for %s (%s); using cache", ticker, exc)
            return pd.read_csv(path, parse_dates=["date"])
        raise RuntimeError(f"fetch failed for {ticker} and no cache exists: {exc}") from exc


def _guess_exchange(ticker: str) -> str:
    if ticker.endswith(".NS"):
        return "NSE"
    if ticker.endswith(".BO"):
        return "BSE"
    return "NASDAQ"


def ensure_stock(session: Session, ticker: str) -> Stock:
    stock = session.get(Stock, ticker)
    if stock is None:
        name, exch, sector = config.DEFAULT_TICKERS.get(ticker, (ticker, _guess_exchange(ticker), None))
        stock = Stock(ticker=ticker, company_name=name, exchange=exch, sector=sector)
        session.add(stock)
        session.flush()
    stock.last_updated = datetime.now()
    return stock


def store_prices(session: Session, ticker: str, df: pd.DataFrame) -> int:
    """Insert only dates not already stored. Returns number of new rows."""
    ensure_stock(session, ticker)
    have = set(session.scalars(select(PriceData.date).where(PriceData.ticker == ticker)))
    new = 0
    for r in df.itertuples(index=False):
        d = pd.Timestamp(r.date).date()
        if d in have:
            continue
        adj = None if pd.isna(r.adj_close) else float(r.adj_close)
        session.add(PriceData(ticker=ticker, date=d, open=float(r.open), high=float(r.high),
                              low=float(r.low), close=float(r.close), volume=int(r.volume),
                              adj_close=adj))
        new += 1
    session.commit()
    return new


def load_prices(session: Session, ticker: str) -> pd.DataFrame:
    rows = session.execute(
        select(PriceData.date, PriceData.open, PriceData.high, PriceData.low,
               PriceData.close, PriceData.volume, PriceData.adj_close)
        .where(PriceData.ticker == ticker).order_by(PriceData.date)
    ).all()
    df = pd.DataFrame(rows, columns=COLS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def ingest_ticker(session: Session, ticker: str, period: str = "2y") -> tuple[pd.DataFrame, int]:
    df = fetch_price_data(ticker, period=period)
    return df, store_prices(session, ticker, df)
