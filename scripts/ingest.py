"""Fetch prices, indicators and news for all tracked tickers.

Run locally (needs internet):  python -m scripts.ingest
Options: --tickers TCS.NS AAPL   --period 2y   --no-news
"""
import argparse
import logging

import config
from db.session import get_session
from ml.indicators import add_indicators, store_indicators
from scripts.init_db import init_db
from services.data_ingestion import ingest_ticker, load_prices
from services.news import fetch_news, store_news


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=list(config.DEFAULT_TICKERS))
    ap.add_argument("--period", default="2y")
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    init_db()
    with get_session() as s:
        for t in args.tickers:
            try:
                df, new = ingest_ticker(s, t, args.period)
                ind = store_indicators(s, t, add_indicators(load_prices(s, t)))
                msg = f"{t:12s} prices: {len(df)} fetched, {new} new | indicators: {ind} new"
                if not args.no_news:
                    name = config.DEFAULT_TICKERS.get(t, (t,))[0]
                    query = config.MARKET_NEWS_QUERIES.get(t, f'"{name}"')
                    msg += f" | news: {store_news(s, t, fetch_news(query))} new"
                print(msg)
            except Exception as exc:
                print(f"{t:12s} FAILED: {exc}")


if __name__ == "__main__":
    main()
