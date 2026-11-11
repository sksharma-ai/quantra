from datetime import datetime

import pytest
from sqlalchemy.orm import Session

from db.models import Base, NewsHeadline, SentimentScore, Stock
from db.session import make_engine
from ml import sentiment as S


def fake_scorer(texts):
    out = []
    for t in texts:
        low = t.lower()
        if "surge" in low or "profit" in low:
            out.append({"label": "Positive", "score": 0.8})
        elif "crash" in low or "loss" in low:
            out.append({"label": "Negative", "score": -0.7})
        else:
            out.append({"label": "Neutral", "score": 0.0})
    return out


@pytest.fixture
def s():
    eng = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as sess:
        sess.add(Stock(ticker="INFY.NS", company_name="Infosys", exchange="NSE"))
        sess.commit()
        rows = [
            ("Infosys profit surges", datetime(2026, 11, 9, 10)),
            ("Infosys shares crash on loss", datetime(2026, 11, 9, 15)),
            ("Infosys holds AGM", datetime(2026, 11, 10, 9)),
            ("Infosys profit beats estimates", datetime(2026, 11, 11, 9)),
            ("Undated headline about profit", None),
        ]
        for i, (h, d) in enumerate(rows):
            sess.add(NewsHeadline(ticker="INFY.NS", headline=h, published_at=d, url=f"http://x/{i}"))
        sess.commit()
        yield sess


def test_preprocess_text():
    assert S.preprocess_text("  Nifty &amp; Sensex\n <b>rally</b>  ") == "Nifty & Sensex rally"
    assert S.preprocess_text(None) == ""


def test_result_from_probs():
    assert S.result_from_probs({"positive": 0.7, "negative": 0.1, "neutral": 0.2}) == pytest.approx(
        {"label": "Positive", "score": 0.6})
    assert S.result_from_probs({"positive": 0.1, "negative": 0.6, "neutral": 0.3})["label"] == "Negative"
    assert S.result_from_probs({"positive": 0.2, "negative": 0.2, "neutral": 0.6})["label"] == "Neutral"


def test_parse_pipeline_output():
    out = [[{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.05}, {"label": "neutral", "score": 0.05}],
           [{"label": "neutral", "score": 0.8}, {"label": "positive", "score": 0.1}, {"label": "negative", "score": 0.1}]]
    r = S.parse_pipeline_output(out)
    assert [x["label"] for x in r] == ["Positive", "Neutral"] and r[0]["score"] == pytest.approx(0.85)


def test_vader_fallback_works():
    r = S.run_vader(["Company reports record profit, great growth", "Terrible loss, fraud scandal", "Meeting on Tuesday"])
    assert [x["label"] for x in r] == ["Positive", "Negative", "Neutral"]
    assert all(-1 <= x["score"] <= 1 for x in r)


def test_get_scorer_falls_back_to_vader(monkeypatch):
    def boom():
        raise OSError("no internet")
    monkeypatch.setattr(S, "_load_pipeline", boom)
    scorer, name = S.get_scorer()
    assert name == "vader" and scorer is S.run_vader


def test_score_headlines_once_each(s):
    assert S.score_headlines(s, fake_scorer, "fake") == 5
    assert S.score_headlines(s, fake_scorer, "fake") == 0        # idempotent
    assert s.query(SentimentScore).count() == 5


def test_daily_aggregate_and_trend(s):
    S.score_headlines(s, fake_scorer, "fake")
    d = S.aggregate_daily_sentiment(s, "INFY.NS")
    assert len(d) == 3                                            # undated headline excluded
    nov9 = d[d["date"] == "2026-11-09"].iloc[0]
    assert nov9["n"] == 2 and nov9["mean_score"] == pytest.approx(0.05)
    assert nov9["pos_share"] == 0.5 and nov9["neg_share"] == 0.5
    t = S.sentiment_trend(d)
    assert t["trend"].iloc[0] == pytest.approx(0.05) and "trend" in t.columns


def test_summary_and_empty(s):
    S.score_headlines(s, fake_scorer, "fake")
    r = S.ticker_summary(s, "INFY.NS")
    assert r["n"] == 4 and (r["positive"], r["neutral"], r["negative"]) == (2, 1, 1)
    assert S.aggregate_daily_sentiment(s, "NOPE").empty
