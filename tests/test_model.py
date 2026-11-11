import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Base, Prediction, Stock
from db.session import make_engine
from ml import model as M
from ml.features import FEATURE_COLUMNS, build_dataset
from tests.conftest import make_prices


@pytest.fixture
def ds(long_prices):
    return build_dataset(long_prices)


def test_walk_forward_is_chronological_and_leak_free(ds):
    wf = M.walk_forward_validate(ds)
    folds = wf["folds"]
    assert len(folds) >= 5
    for f in folds:
        assert f["n_train"] == 252 and f["n_test"] == 63
        assert f["train_end"] < f["test_start"]          # train strictly before test
    for a, b in zip(folds, folds[1:]):
        assert a["test_end"] < b["test_end"]               # folds move forward in time
    last_label_date = str(M.labeled(ds)["date"].iloc[-1].date())
    assert folds[-1]["test_end"] == last_label_date        # latest fold = most recent data


def test_too_little_data_raises(prices):
    with pytest.raises(ValueError):
        M.walk_forward_validate(build_dataset(prices))


def test_random_walk_not_predictable(ds):
    """Guard against leakage: on pure noise, accuracy must stay near 50%."""
    assert M.walk_forward_validate(ds)["mean"]["accuracy"] < 0.62


def test_model_learns_real_signal():
    """Sanity: if a signal exists (mean-reverting returns) the pipeline must find it."""
    d = build_dataset(make_prices(700, seed=7, rho=-0.9, vol=0.012))
    assert M.walk_forward_validate(d)["mean"]["accuracy"] > 0.70


def test_predict_and_explanation(ds):
    model = M.train_final(ds)
    r = M.predict_latest(model, ds)
    assert r["direction"] in ("UP", "DOWN") and 0.5 <= r["confidence"] <= 1
    assert len(r["drivers"]) == 5
    assert all(d["feature"] in FEATURE_COLUMNS for d in r["drivers"])
    assert r["direction"] in r["reasoning"] and "%" in r["reasoning"]
    # drivers must be sorted by size of influence
    mags = [abs(d["contribution"]) for d in r["drivers"]]
    assert mags == sorted(mags, reverse=True)
    # explanation is faithful: contributions + bias reproduce the model probability
    assert 1 / (1 + np.exp(-r["log_odds"])) == pytest.approx(r["p_up"], abs=1e-4)


def test_save_load_roundtrip(ds, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MODEL_DIR", tmp_path)
    model = M.train_final(ds)
    M.save_model(model, "^NSEI")
    again = M.load_model("^NSEI")
    assert M.predict_latest(again, ds)["p_up"] == pytest.approx(M.predict_latest(model, ds)["p_up"], abs=1e-6)


def test_store_prediction_upsert(ds):
    eng = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(Stock(ticker="TCS.NS", company_name="TCS", exchange="NSE")); s.commit()
        r = M.predict_latest(M.train_final(ds), ds)
        M.store_prediction(s, "TCS.NS", r)
        M.store_prediction(s, "TCS.NS", r)
        rows = s.scalars(select(Prediction)).all()
        assert len(rows) == 1 and rows[0].direction == r["direction"]
        assert rows[0].model_version.startswith("xgb-v1")


# ---------------------------------------------------------------- Step 3b
def test_edge_test_noise_vs_perfect():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    assert not M.edge_test(y, rng.uniform(0, 1, 800))["significant_edge"]
    assert M.edge_test(y, np.where(y == 1, 0.8, 0.2))["significant_edge"]


def test_calibrator_shrinks_overconfident_noise():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 900)
    p = np.clip(rng.normal(0.5, 0.15, 900), 0.02, 0.98)  # looks confident, carries no signal
    cal = M.fit_calibrator(p, y)
    assert abs(M.apply_calibration(cal, 0.9) - 0.5) < 0.12   # a raw 90% is NOT believed
    assert cal["significant_edge"] is False


def test_calibrator_never_inverts_a_bad_model():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 900)
    p = np.where(y == 1, rng.uniform(0.1, 0.5, 900), rng.uniform(0.5, 0.9, 900))  # anti-predictive
    cal = M.fit_calibrator(p, y)
    assert cal["a"] == 0.0


def test_calibrator_trusts_real_signal():
    d = build_dataset(make_prices(700, seed=7, rho=-0.9, vol=0.012))
    wf = M.walk_forward_validate(d)
    cal = M.fit_calibrator(wf["oof_proba"], wf["oof_target"])
    assert cal["a"] > 0 and cal["significant_edge"]
    r = M.predict_latest(M.train_final(d), d, cal)
    assert r["confidence_label"] in ("LOW", "MODERATE", "HIGH") and 0 <= r["confidence"] <= 1


def test_unproven_model_is_always_low(ds):
    wf = M.walk_forward_validate(ds)
    cal = M.fit_calibrator(wf["oof_proba"], wf["oof_target"])
    r = M.predict_latest(M.train_final(ds), ds, cal)
    assert r["confidence_label"] == "LOW" or cal["significant_edge"]
    assert M.confidence_label(0.9, significant_edge=False) == "LOW"


def test_calibration_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MODEL_DIR", tmp_path)
    cal = {"a": 0.2, "b": 0.1, "n": 5, "significant_edge": False}
    M.save_calibration(cal, "TCS.NS")
    assert M.load_calibration("TCS.NS") == cal and M.load_calibration("NOPE") is None
