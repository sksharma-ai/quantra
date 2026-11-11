import pandas as pd
import pytest
from sqlalchemy.orm import Session

from db.models import Base
from db.session import make_engine
from ml.indicators import add_indicators, store_indicators
from services import data_ingestion as di
from services import news as nw


@pytest.fixture
def s():
    eng = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as sess:
        yield sess


def test_normalize_multiindex_and_clamp(prices):
    raw = prices.rename(columns=str.title).rename(columns={"Adj_Close": "Adj Close"}).set_index("Date")
    raw.index.name = "Date"
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["TCS.NS"]])  # new yfinance layout
    raw.iloc[5, raw.columns.get_loc(("High", "TCS.NS"))] = 0.01  # impossible high
    out = di.normalize_prices(raw)
    assert (out["high"] >= out["open"]).all() and (out["low"] <= out["open"]).all()
    assert list(out.columns) == di.COLS and len(out) == len(prices)


def test_normalize_drops_bad_rows(prices):
    bad = prices.copy()
    bad.loc[3, "close"] = None
    bad.loc[4, "open"] = -1
    assert len(di.normalize_prices(bad.set_index("date"))) == len(prices) - 2


def test_store_prices_idempotent(s, prices):
    assert di.store_prices(s, "TCS.NS", prices) == len(prices)
    assert di.store_prices(s, "TCS.NS", prices) == 0
    assert len(di.load_prices(s, "TCS.NS")) == len(prices)


def test_fetch_uses_cache_on_failure(monkeypatch, tmp_path, prices):
    monkeypatch.setattr(di, "CACHE_DIR", tmp_path)
    prices.to_csv(di._cache_path("TCS.NS"), index=False)

    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(di, "_download", boom)
    assert len(di.fetch_price_data("TCS.NS")) == len(prices)
    with pytest.raises(RuntimeError):
        di.fetch_price_data("UNKNOWN")  # no cache, no network


def test_store_indicators(s, prices):
    di.store_prices(s, "TCS.NS", prices)
    n = store_indicators(s, "TCS.NS", add_indicators(di.load_prices(s, "TCS.NS")))
    assert n > 250 and store_indicators(s, "TCS.NS", add_indicators(di.load_prices(s, "TCS.NS"))) == 0


PAYLOAD = {"articles": [
    {"title": "Infosys wins big deal", "url": "http://a/1", "publishedAt": "2026-11-10T08:30:00Z", "source": {"name": "ET"}},
    {"title": "[Removed]", "url": "http://a/2", "publishedAt": "2026-11-10T08:30:00Z", "source": {"name": "X"}},
    {"title": "", "url": "http://a/3", "source": {}},
    {"title": "Nifty ends flat", "url": "http://a/4", "publishedAt": None, "source": {"name": "Mint"}},
]}


def test_parse_articles():
    items = nw.parse_articles(PAYLOAD)
    assert [i["url"] for i in items] == ["http://a/1", "http://a/4"]
    assert items[0]["published_at"].tzinfo is None and items[1]["published_at"] is None


def test_store_news_dedup(s):
    items = nw.parse_articles(PAYLOAD)
    assert nw.store_news(s, "INFY.NS", items) == 2
    assert nw.store_news(s, "INFY.NS", items) == 0


def test_news_no_key_no_cache_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(nw, "CACHE_DIR", tmp_path)
    assert nw.fetch_news("anything", api_key="") == []
