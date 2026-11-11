"""M1: news headline ingestion (NewsAPI) with JSON cache fallback."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from db.models import NewsHeadline
from services.data_ingestion import ensure_stock

log = logging.getLogger(__name__)

NEWS_URL = "https://newsapi.org/v2/everything"
CACHE_DIR = Path("data/cache")


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"news_{re.sub(r'[^A-Za-z0-9]+', '_', key)[:60]}.json"


def parse_articles(payload: dict) -> list[dict]:
    """NewsAPI JSON -> list of {headline, source, published_at, url}."""
    out = []
    for a in payload.get("articles", []):
        title, url = (a.get("title") or "").strip(), a.get("url")
        if not title or not url or title == "[Removed]":
            continue
        ts = pd.to_datetime(a.get("publishedAt"), utc=True, errors="coerce")
        out.append({
            "headline": title,
            "source": (a.get("source") or {}).get("name"),
            "published_at": None if pd.isna(ts) else ts.tz_convert(None).to_pydatetime(),
            "url": url,
        })
    return out


def fetch_news(query: str, api_key: str | None = None, page_size: int = 50,
               days_back: int = 7) -> list[dict]:
    """Fetch headlines for a query. Falls back to cache if key missing or call fails."""
    import requests

    api_key = config.NEWSAPI_KEY if api_key is None else api_key
    cache = _cache_path(query)
    try:
        if not api_key:
            raise RuntimeError("NEWSAPI_KEY not set")
        params = {
            "q": query, "language": "en", "sortBy": "publishedAt", "pageSize": page_size,
            "from": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
            "apiKey": api_key,
        }
        resp = requests.get(NEWS_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload), encoding="utf-8")
        return parse_articles(payload)
    except Exception as exc:
        msg = str(exc).replace(api_key, "***") if api_key else str(exc)  # never log the key
        if cache.exists():
            log.warning("news fetch failed (%s); using cache", msg)
            return parse_articles(json.loads(cache.read_text(encoding="utf-8")))
        log.warning("news fetch failed (%s) and no cache", msg)
        return []


def store_news(session: Session, ticker: str | None, items: list[dict]) -> int:
    """Insert headlines whose URL is not stored yet. Returns number of new rows."""
    if ticker:
        ensure_stock(session, ticker)
    have = set(session.scalars(select(NewsHeadline.url)))
    new = 0
    for it in items:
        if it["url"] in have:
            continue
        session.add(NewsHeadline(ticker=ticker, headline=it["headline"], source=it["source"],
                                 published_at=it["published_at"], url=it["url"]))
        have.add(it["url"])
        new += 1
    session.commit()
    return new
