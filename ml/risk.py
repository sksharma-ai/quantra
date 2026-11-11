"""M5: market crash risk score (0-100) = volatility + anomaly (Isolation Forest) + news sentiment.

All pieces use only past data. Sentiment only exists for the last ~month of news, so historical
scores use volatility + anomaly (weights renormalised); the latest score adds sentiment when available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import RiskScore
from services.data_ingestion import ensure_stock

WEIGHTS = {"volatility": 0.45, "anomaly": 0.30, "sentiment": 0.25}
LOW_MAX, HIGH_MIN = 35.0, 65.0   # score < 35 LOW, < 65 MED, else HIGH
TRAIN_WINDOW = 252               # one trading year
REFIT_EVERY = 21                 # refit Isolation Forest monthly
MIN_ROWS = 80                    # minimum price rows for a volatility score


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x)) and not pd.isna(x)


# ------------------------------------------------------------- volatility
def calc_volatility_score(prices: pd.DataFrame) -> pd.DataFrame:
    """60% = how high 20-day volatility ranks vs the past year; 40% = depth below the 60-day high."""
    c = prices["close"]
    r = c.pct_change()
    ann = r.rolling(20).std() * np.sqrt(252) * 100
    a = ann.dropna()
    pct = a.rolling(TRAIN_WINDOW, min_periods=60).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True)
    pct = pct.reindex(ann.index)
    dd = c / c.rolling(60, min_periods=20).max() - 1
    dd_score = (-dd / 0.15).clip(0, 1) * 100   # 0% below high -> 0, 15%+ below -> 100
    return pd.DataFrame({
        "vol_annual_pct": ann, "vol_percentile": pct, "drawdown_60": dd,
        "drawdown_score": dd_score, "volatility_component": 0.6 * pct + 0.4 * dd_score,
    })


# ---------------------------------------------------------------- anomaly
def _anomaly_features(prices: pd.DataFrame) -> pd.DataFrame:
    c, r = prices["close"], prices["close"].pct_change()
    vol_ratio = (prices["volume"] / prices["volume"].rolling(20).mean()).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({
        "ret_1": r, "ret_5": c.pct_change(5), "vol_20": r.rolling(20).std(),
        "vol_ratio": vol_ratio.fillna(1.0),   # index tickers may report zero volume
        "gap": prices["open"] / c.shift(1) - 1,
        "dd_60": c / c.rolling(60, min_periods=20).max() - 1,
    })


def run_isolation_forest(prices: pd.DataFrame, train_window: int = TRAIN_WINDOW,
                         refit_every: int = REFIT_EVERY) -> pd.Series:
    """For each day: how unusual is it versus the previous `train_window` days? (0-100)
    100 = more unusual than every day in the training year. Model sees only data before the day."""
    f = _anomaly_features(prices).dropna()
    X = f.to_numpy()
    out = pd.Series(np.nan, index=f.index)
    t = train_window
    while t < len(f):
        end = min(t + refit_every, len(f))
        train = X[t - train_window:t]
        iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=1).fit(train)
        tr_scores = iso.score_samples(train)        # higher = more normal
        new_scores = iso.score_samples(X[t:end])
        out.iloc[t:end] = [(tr_scores > s).mean() * 100 for s in new_scores]
        t = end
    return out.reindex(prices.index)


def calc_anomaly_component(prices: pd.DataFrame) -> pd.DataFrame:
    """Crash risk cares about DOWNSIDE oddities: halve the score when the 5-day move is up."""
    raw = run_isolation_forest(prices)
    ret5 = prices["close"].pct_change(5)
    factor = np.where(ret5 < 0, 1.0, 0.5)
    return pd.DataFrame({"anomaly_raw": raw, "ret_5": ret5, "anomaly_component": raw * factor})


# -------------------------------------------------------------- sentiment
def calc_sentiment_component(daily: pd.DataFrame | None, lookback_days: int = 7,
                             full_weight_n: int = 10) -> dict:
    """Average news tone (-1..+1) over the last week -> risk 0-100 (negative news = high risk).
    Few headlines -> shrunk toward neutral (50). No news -> component unavailable (None)."""
    if daily is None or len(daily) == 0:
        return {"component": None, "n": 0, "mean": None}
    end = daily["date"].max()
    recent = daily[daily["date"] > end - pd.Timedelta(days=lookback_days)]
    n = int(recent["n"].sum())
    if n == 0:
        return {"component": None, "n": 0, "mean": None}
    mean = float((recent["mean_score"] * recent["n"]).sum() / n)   # headline-weighted
    raw = (1 - mean) / 2 * 100
    w = min(1.0, n / full_weight_n)
    return {"component": 50 + w * (raw - 50), "n": n, "mean": mean}


# ---------------------------------------------------------------- combine
def combine_signals(vol, sent, anom, weights: dict = WEIGHTS) -> dict | None:
    """Weighted average over the components that exist (weights renormalised). None if none exist."""
    parts = {"volatility": vol, "sentiment": sent, "anomaly": anom}
    used = {k: weights[k] for k, v in parts.items() if _ok(v)}
    if not used:
        return None
    total = sum(used.values())
    score = sum(weights[k] / total * float(parts[k]) for k in used)
    return {"score": float(np.clip(score, 0, 100)), "weights": {k: weights[k] / total for k in used}}


def classify_risk(score: float) -> str:
    return "LOW" if score < LOW_MAX else "MED" if score < HIGH_MIN else "HIGH"


# ---------------------------------------------------------------- compute
def risk_history(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily score from volatility + anomaly (no sentiment: no historic news)."""
    v = calc_volatility_score(prices)["volatility_component"]
    a = calc_anomaly_component(prices)["anomaly_component"]
    wv, wa = WEIGHTS["volatility"], WEIGHTS["anomaly"]
    score = pd.Series(np.nan, index=prices.index)
    both = v.notna() & a.notna()
    score[both] = ((wv * v + wa * a) / (wv + wa))[both]
    only_v = v.notna() & a.isna()
    score[only_v] = v[only_v]
    out = pd.DataFrame({"date": prices["date"], "volatility_component": v,
                        "anomaly_component": a, "score": score})
    out["level"] = out["score"].map(lambda s: classify_risk(s) if _ok(s) else None)
    return out


def compute_risk(prices: pd.DataFrame, sentiment_daily: pd.DataFrame | None = None) -> dict:
    """Latest crash risk with component breakdown and plain-language reasoning (FR-05, FR-10)."""
    if len(prices) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} price rows, have {len(prices)}")
    vol = calc_volatility_score(prices).iloc[-1]
    an = calc_anomaly_component(prices).iloc[-1]
    sent = calc_sentiment_component(sentiment_daily)
    vol_c = float(vol["volatility_component"])
    anom_c = float(an["anomaly_component"]) if _ok(an["anomaly_component"]) else None

    combo = combine_signals(vol_c, sent["component"], anom_c)
    score = combo["score"]
    level = classify_risk(score)

    parts = [f"Volatility {vol_c:.0f}/100: 20-day volatility ({vol['vol_annual_pct']:.1f}% annualised) is "
             f"higher than {vol['vol_percentile']:.0f}% of the past year, and price is "
             f"{abs(vol['drawdown_60']) * 100:.1f}% below its 60-day high."]
    if anom_c is not None:
        note = "" if an["ret_5"] < 0 else " (halved because the 5-day move is upward)"
        parts.append(f"Anomaly {anom_c:.0f}/100: Isolation Forest finds recent price and volume behaviour more "
                     f"unusual than {an['anomaly_raw']:.0f}% of the past year{note}.")
    else:
        parts.append("Anomaly: not enough history yet, not used.")
    if sent["component"] is not None:
        parts.append(f"Sentiment {sent['component']:.0f}/100: average news tone {sent['mean']:+.2f} "
                     f"across {sent['n']} headlines in the last 7 days.")
    else:
        parts.append("Sentiment: no recent news, not used.")
    reasoning = f"Crash risk {score:.0f}/100 ({level}). AI-generated simulation, not advice. " + " ".join(parts)

    return {
        "as_of": pd.Timestamp(prices["date"].iloc[-1]).date(),
        "score": score, "level": level, "volatility_component": vol_c,
        "sentiment_component": sent["component"], "anomaly_component": anom_c,
        "weights_used": combo["weights"], "n_headlines": sent["n"], "reasoning": reasoning,
    }


# ------------------------------------------------------------- validation
def evaluate_risk(prices: pd.DataFrame, hist: pd.DataFrame, horizon: int = 10,
                  drop_threshold: float = 0.05) -> dict:
    """Did HIGH-risk days really precede bigger drops? Forward drop = worst close in the next
    `horizon` days vs today. Reported honestly, whatever it shows."""
    c = prices["close"]
    fwd_min = c.rolling(horizon).min().shift(-horizon)        # min over days t+1 .. t+horizon
    fwd_dd = ((c - fwd_min) / c).clip(lower=0).rename("fwd_dd")
    df = hist[["score", "level"]].join(fwd_dd).dropna()
    rho, p = spearmanr(df["score"], df["fwd_dd"])
    base = float((df["fwd_dd"] > drop_threshold).mean())
    levels = {}
    for lvl in ("LOW", "MED", "HIGH"):
        sub = df[df["level"] == lvl]
        levels[lvl] = {"n": int(len(sub)),
                       "mean_fwd_drop": float(sub["fwd_dd"].mean()) if len(sub) else None,
                       "drop_rate": float((sub["fwd_dd"] > drop_threshold).mean()) if len(sub) else None}
    high_rate = levels["HIGH"]["drop_rate"]
    return {"n": int(len(df)), "horizon_days": horizon, "drop_threshold": drop_threshold,
            "spearman_rho": float(rho), "spearman_p": float(p), "base_drop_rate": base,
            "by_level": levels,
            "lift_high_vs_base": (high_rate / base) if (high_rate is not None and base > 0) else None}


# ------------------------------------------------------------ persistence
def store_risk_score(session: Session, ticker: str, res: dict) -> int:
    """One row per (ticker, date): update if present."""
    ensure_stock(session, ticker)
    row = session.scalars(select(RiskScore).where(
        RiskScore.ticker == ticker, RiskScore.score_date == res["as_of"])).first()
    if row is None:
        row = RiskScore(ticker=ticker, score_date=res["as_of"])
        session.add(row)
    row.crash_risk_score = res["score"]
    row.volatility_component = res["volatility_component"]
    row.sentiment_component = res["sentiment_component"]
    row.anomaly_component = res["anomaly_component"]
    row.risk_level = res["level"]
    session.commit()
    return row.risk_id


def store_risk_history(session: Session, ticker: str, hist: pd.DataFrame) -> int:
    """Insert historic daily scores for dates not stored yet. Returns number of new rows."""
    ensure_stock(session, ticker)
    have = set(session.scalars(select(RiskScore.score_date).where(RiskScore.ticker == ticker)))
    new = 0
    for r in hist.dropna(subset=["score"]).itertuples(index=False):
        d = pd.Timestamp(r.date).date()
        if d in have:
            continue
        session.add(RiskScore(
            ticker=ticker, score_date=d, crash_risk_score=float(r.score),
            volatility_component=float(r.volatility_component),
            anomaly_component=None if pd.isna(r.anomaly_component) else float(r.anomaly_component),
            sentiment_component=None, risk_level=r.level))
        new += 1
    session.commit()
    return new


def load_risk_history(session: Session, ticker: str) -> pd.DataFrame:
    rows = session.execute(
        select(RiskScore.score_date, RiskScore.crash_risk_score, RiskScore.volatility_component,
               RiskScore.sentiment_component, RiskScore.anomaly_component, RiskScore.risk_level)
        .where(RiskScore.ticker == ticker).order_by(RiskScore.score_date)).all()
    df = pd.DataFrame(rows, columns=["date", "score", "volatility_component", "sentiment_component",
                                     "anomaly_component", "level"])
    df["date"] = pd.to_datetime(df["date"])
    return df
