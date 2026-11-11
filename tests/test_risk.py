import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Base, RiskScore, Stock
from db.session import make_engine
from ml import risk as R


def series_from_returns(ret, start="2023-01-02"):
    close = 100 * np.exp(np.cumsum(ret))
    n = len(ret)
    rng = np.random.default_rng(5)
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n))
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n), "open": open_,
        "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
        "close": close, "volume": rng.integers(1_000_000, 2_000_000, n), "adj_close": close})


@pytest.fixture
def crash_prices():
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0004, 0.005, 400)
    crash = rng.normal(-0.02, 0.03, 15)
    return series_from_returns(np.r_[calm, crash])


@pytest.fixture
def calm_prices():
    return series_from_returns(np.random.default_rng(1).normal(0.0004, 0.005, 415))


def test_volatility_bounded_and_reacts_to_crash(crash_prices, calm_prices):
    v = R.calc_volatility_score(crash_prices)["volatility_component"].dropna()
    assert v.between(0, 100).all()
    assert v.iloc[-1] > 85                                   # crash regime
    assert R.calc_volatility_score(calm_prices)["volatility_component"].iloc[-1] < v.iloc[-1] - 30


def test_anomaly_flags_shock_and_has_no_lookahead(crash_prices):
    a = R.run_isolation_forest(crash_prices)
    assert a.dropna().between(0, 100).all()
    assert a.iloc[-5:].max() > 90                            # crash days are unusual
    cut = 360
    changed = crash_prices.copy()
    changed.loc[cut:, ["open", "high", "low", "close"]] *= 2.5
    b = R.run_isolation_forest(changed)
    pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])


def test_anomaly_halved_when_move_is_up():
    p = series_from_returns(np.r_[np.random.default_rng(2).normal(0, 0.005, 330), [0.08] * 3])
    out = R.calc_anomaly_component(p).iloc[-1]
    assert out["ret_5"] > 0 and out["anomaly_component"] == pytest.approx(out["anomaly_raw"] * 0.5)


def test_combine_signals_weights():
    full = R.combine_signals(80, 40, 60)
    assert full["score"] == pytest.approx(0.45 * 80 + 0.25 * 40 + 0.30 * 60)
    no_sent = R.combine_signals(80, None, 60)
    assert no_sent["score"] == pytest.approx((0.45 * 80 + 0.30 * 60) / 0.75)
    assert sum(no_sent["weights"].values()) == pytest.approx(1.0)
    assert R.combine_signals(None, float("nan"), None) is None
    assert R.combine_signals(500, None, None)["score"] == 100


def test_classify_risk():
    assert [R.classify_risk(x) for x in (0, 34.9, 35, 64.9, 65, 100)] == ["LOW", "LOW", "MED", "MED", "HIGH", "HIGH"]


def daily(rows):
    return pd.DataFrame(rows, columns=["date", "mean_score", "n"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_sentiment_component():
    assert R.calc_sentiment_component(None)["component"] is None
    assert R.calc_sentiment_component(daily([]))["component"] is None
    bad = R.calc_sentiment_component(daily([("2026-11-10", -0.8, 20)]))
    good = R.calc_sentiment_component(daily([("2026-11-10", 0.8, 20)]))
    assert bad["component"] == pytest.approx(90) and good["component"] == pytest.approx(10)
    few = R.calc_sentiment_component(daily([("2026-11-10", -0.8, 2)]))
    assert 50 < few["component"] < bad["component"]           # shrunk toward neutral
    old = R.calc_sentiment_component(daily([("2026-10-01", -0.9, 5), ("2026-11-10", 0.2, 10)]))
    assert old["n"] == 10                                      # week-old news ignored


def test_compute_risk_full_and_missing_parts(crash_prices):
    res = R.compute_risk(crash_prices, daily([("2026-11-10", -0.5, 15)]))
    assert 0 <= res["score"] <= 100 and res["level"] in ("LOW", "MED", "HIGH")
    assert res["sentiment_component"] is not None and "/100" in res["reasoning"]
    assert res["level"] == "HIGH"                              # crash regime must read as HIGH
    res2 = R.compute_risk(crash_prices)
    assert res2["sentiment_component"] is None and "no recent news" in res2["reasoning"]
    assert "sentiment" not in res2["weights_used"]


def test_compute_risk_short_history(calm_prices):
    short = calm_prices.iloc[:120]                             # anomaly needs ~270 rows
    res = R.compute_risk(short)
    assert res["anomaly_component"] is None and "not enough history" in res["reasoning"]
    with pytest.raises(ValueError):
        R.compute_risk(calm_prices.iloc[:50])


def test_risk_predicts_drops_on_regime_data():
    """Volatility clusters in real markets; on such data HIGH-risk days must precede bigger drops."""
    rng = np.random.default_rng(11)
    vols = np.repeat([0.005, 0.03, 0.006, 0.028, 0.005, 0.03, 0.006, 0.027, 0.005, 0.03, 0.006, 0.026, 0.005, 0.03], 70)
    p = series_from_returns(rng.normal(0.0002, vols))
    hist = R.risk_history(p)
    ev = R.evaluate_risk(p, hist)
    assert ev["spearman_rho"] > 0.15 and ev["spearman_p"] < 0.01
    assert ev["by_level"]["HIGH"]["mean_fwd_drop"] > ev["by_level"]["LOW"]["mean_fwd_drop"]
    assert ev["lift_high_vs_base"] > 1.0


@pytest.fixture
def s():
    eng = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as sess:
        yield sess


def test_store_latest_and_history(s, crash_prices):
    hist = R.risk_history(crash_prices)
    n = R.store_risk_history(s, "^NSEI", hist)
    assert n == hist["score"].notna().sum() and R.store_risk_history(s, "^NSEI", hist) == 0
    res = R.compute_risk(crash_prices, daily([("2026-11-10", -0.5, 15)]))
    R.store_risk_score(s, "^NSEI", res)
    R.store_risk_score(s, "^NSEI", res)
    rows = s.scalars(select(RiskScore).where(RiskScore.score_date == res["as_of"])).all()
    assert len(rows) == 1 and rows[0].sentiment_component is not None   # overwrote the history row
    assert R.load_risk_history(s, "^NSEI")["date"].is_monotonic_increasing
