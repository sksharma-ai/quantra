"""M4: news sentiment. FinBERT (primary) with VADER fallback.

A scorer is any function: list[str] -> list[{"label": Positive|Neutral|Negative, "score": -1..+1}].
score > 0 means positive news. Scorers are injectable so tests do not need to download a model.
"""
from __future__ import annotations

import html
import logging
import re
from functools import lru_cache
from typing import Callable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import NewsHeadline, SentimentScore

log = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
Scorer = Callable[[list[str]], list[dict]]
LABELS = ("Positive", "Neutral", "Negative")


def preprocess_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------ FinBERT
def result_from_probs(probs: dict) -> dict:
    """{'positive': p, 'negative': n, 'neutral': u} -> label (argmax) + signed score (p - n)."""
    p, n, u = (float(probs.get(k, 0.0)) for k in ("positive", "negative", "neutral"))
    label = max((("Positive", p), ("Negative", n), ("Neutral", u)), key=lambda kv: kv[1])[0]
    return {"label": label, "score": p - n}


def parse_pipeline_output(out: list) -> list[dict]:
    """HuggingFace text-classification output with top_k=None: one list of {label, score} per text."""
    return [result_from_probs({d["label"].lower(): d["score"] for d in item}) for item in out]


@lru_cache(maxsize=1)
def _load_pipeline():
    from transformers import pipeline  # heavy import, only when actually needed

    return pipeline("text-classification", model=FINBERT_MODEL, top_k=None)


def run_finbert(texts: list[str], batch_size: int = 16) -> list[dict]:
    if not texts:
        return []
    out = _load_pipeline()(list(texts), batch_size=batch_size, truncation=True, max_length=128)
    return parse_pipeline_output(out)


# -------------------------------------------------------------------- VADER
def run_vader(texts: list[str]) -> list[dict]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    an = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = an.polarity_scores(t)["compound"]
        out.append({"label": "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral",
                    "score": float(c)})
    return out


def get_scorer() -> tuple[Scorer, str]:
    """FinBERT if it loads (needs torch + internet on first run), else VADER."""
    try:
        _load_pipeline()
        return run_finbert, "finbert"
    except Exception as exc:
        log.warning("FinBERT unavailable (%s: %s); falling back to VADER", type(exc).__name__, exc)
        return run_vader, "vader"


# ------------------------------------------------------------------ storing
def score_headlines(session: Session, scorer: Scorer, model_name: str, chunk: int = 64) -> int:
    """Score every headline that has no sentiment yet. Returns number scored."""
    todo = session.execute(
        select(NewsHeadline.news_id, NewsHeadline.headline)
        .outerjoin(SentimentScore, SentimentScore.news_id == NewsHeadline.news_id)
        .where(SentimentScore.sent_id.is_(None))
    ).all()
    done = 0
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        results = scorer([preprocess_text(h) for _, h in part])
        for (news_id, _), r in zip(part, results):
            session.add(SentimentScore(news_id=news_id, sentiment_label=r["label"],
                                       sentiment_score=float(r["score"]), model_used=model_name))
        session.commit()
        done += len(part)
    return done


# -------------------------------------------------------------- aggregation
def _scored_frame(session: Session, ticker: str) -> pd.DataFrame:
    rows = session.execute(
        select(NewsHeadline.published_at, SentimentScore.sentiment_label, SentimentScore.sentiment_score)
        .join(SentimentScore, SentimentScore.news_id == NewsHeadline.news_id)
        .where(NewsHeadline.ticker == ticker, NewsHeadline.published_at.is_not(None))
    ).all()
    df = pd.DataFrame(rows, columns=["published_at", "label", "score"])
    df["date"] = pd.to_datetime(df["published_at"]).dt.normalize()
    return df


def aggregate_daily_sentiment(session: Session, ticker: str) -> pd.DataFrame:
    """One row per day: mean score (-1..+1), headline count, share positive / negative."""
    df = _scored_frame(session, ticker)
    cols = ["date", "mean_score", "n", "pos_share", "neg_share"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    g = df.groupby("date")
    out = pd.DataFrame({
        "mean_score": g["score"].mean(),
        "n": g["score"].size(),
        "pos_share": g["label"].apply(lambda s: (s == "Positive").mean()),
        "neg_share": g["label"].apply(lambda s: (s == "Negative").mean()),
    }).reset_index()
    return out[cols]


def sentiment_trend(daily: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = daily.sort_values("date").copy()
    out["trend"] = out["mean_score"].rolling(window, min_periods=1).mean()
    return out


def ticker_summary(session: Session, ticker: str) -> dict:
    df = _scored_frame(session, ticker)
    counts = df["label"].value_counts()
    return {
        "n": int(len(df)),
        "mean_score": float(df["score"].mean()) if len(df) else None,
        **{k.lower(): int(counts.get(k, 0)) for k in LABELS},
    }
