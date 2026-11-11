"""M3: XGBoost next-day direction model, walk-forward validation, per-prediction explanation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import binomtest
from sqlalchemy import select
from sqlalchemy.orm import Session
from xgboost import DMatrix, XGBClassifier

from db.models import Prediction
from ml.features import FEATURE_COLUMNS, N_LAGS

TRAIN_WINDOW = 252  # trading days (about 1 year), per synopsis
TEST_WINDOW = 63    # trading days (about 1 quarter), per synopsis
MODEL_DIR = Path("models_saved")
MODEL_VERSION = "xgb-v1"

# Shallow, regularised trees: daily data is small and very noisy, deep trees just memorise.
PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
    eval_metric="logloss", random_state=42, n_jobs=1,
)

# feature -> (human label, format kind). kinds: num = plain, pct = signed %, abs = unsigned %
FEATURE_META = {
    "rsi": ("RSI(14)", "num"),
    "macd": ("MACD line", "num"),
    "macd_signal": ("MACD signal line", "num"),
    "macd_hist": ("MACD histogram", "num"),
    "bb_position": ("Position in Bollinger band (0=lower, 1=upper)", "num"),
    "bb_width": ("Bollinger band width", "abs"),
    "close_to_sma7": ("Price vs 7-day average", "pct"),
    "close_to_sma21": ("Price vs 21-day average", "pct"),
    "close_to_sma50": ("Price vs 50-day average", "pct"),
    "close_to_ema9": ("Price vs 9-day EMA", "pct"),
    "close_to_ema21": ("Price vs 21-day EMA", "pct"),
    "volume_change": ("Volume change vs previous day", "pct"),
    "daily_return": ("Latest daily return", "pct"),
    "overnight_gap": ("Overnight gap", "pct"),
    "std_7": ("7-day volatility", "abs"),
    "std_30": ("30-day volatility", "abs"),
}
FEATURE_META.update({f"ret_lag_{k}": (f"Return {k} day(s) earlier", "pct") for k in range(1, N_LAGS + 1)})


def labeled(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset[dataset["target"].notna()].reset_index(drop=True)


def new_model() -> XGBClassifier:
    return XGBClassifier(**PARAMS)


# ---------------------------------------------------------------- validation
def _metrics(y_true, proba, baseline_class) -> dict:
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, proba)) if len(set(y_true)) == 2 else None
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "baseline_accuracy": float(np.mean(y_true == baseline_class)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": auc,
    }


def walk_forward_validate(dataset: pd.DataFrame, train_window: int = TRAIN_WINDOW,
                          test_window: int = TEST_WINDOW, step: int | None = None) -> dict:
    """Rolling chronological folds: train on the past `train_window` days, test on the next
    `test_window`. The last fold always ends at the most recent labelled day. No shuffling."""
    data = labeled(dataset)
    step = step or test_window
    if len(data) < train_window + test_window:
        raise ValueError(f"need {train_window + test_window} labelled rows, have {len(data)}")

    spans, test_end = [], len(data)
    while test_end - test_window - train_window >= 0:
        spans.append((test_end - test_window - train_window, test_end - test_window, test_end))
        test_end -= step
    spans.reverse()  # chronological order

    folds, oof_p, oof_y = [], [], []
    for i, (tr0, te0, te1) in enumerate(spans, 1):
        train, test = data.iloc[tr0:te0], data.iloc[te0:te1]
        model = new_model().fit(train[FEATURE_COLUMNS], train["target"].astype(int))
        proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        base = int(train["target"].mean() >= 0.5)
        y_test = test["target"].astype(int).to_numpy()
        oof_p.extend(proba.tolist())
        oof_y.extend(y_test.tolist())
        m = _metrics(y_test, proba, base)
        m.update({
            "fold": i, "n_train": len(train), "n_test": len(test),
            "train_start": str(train["date"].iloc[0].date()), "train_end": str(train["date"].iloc[-1].date()),
            "test_start": str(test["date"].iloc[0].date()), "test_end": str(test["date"].iloc[-1].date()),
        })
        folds.append(m)

    keys = ["accuracy", "baseline_accuracy", "precision", "recall", "f1"]
    mean = {k: float(np.mean([f[k] for f in folds])) for k in keys}
    aucs = [f["roc_auc"] for f in folds if f["roc_auc"] is not None]
    mean["roc_auc"] = float(np.mean(aucs)) if aucs else None
    return {"folds": folds, "mean": mean, "latest_fold": folds[-1],
            "pooled": edge_test(oof_y, oof_p), "oof_proba": oof_p, "oof_target": oof_y}


# ------------------------------------------------- edge test + calibration
def edge_test(y_true, proba) -> dict:
    """Does the model beat 'always guess the majority class' by more than luck?
    One-sided binomial test on pooled out-of-fold predictions."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= 0.5).astype(int)
    n, correct = len(y), int((pred == y).sum())
    baseline = float(max(y.mean(), 1 - y.mean()))
    p = float(binomtest(correct, n, baseline, alternative="greater").pvalue)
    return {"n": n, "pooled_accuracy": correct / n, "baseline_accuracy": baseline,
            "p_value": p, "significant_edge": bool(p < 0.05)}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_calibrator(oof_proba, oof_target) -> dict:
    """Platt scaling on out-of-fold predictions: maps raw model score -> honest probability.
    A negative slope (model worse than chance) is NOT trusted: slope is set to 0 so the
    output collapses to the base rate instead of inverting the model."""
    y = np.asarray(oof_target).astype(int)
    lr = LogisticRegression(C=1.0).fit(_logit(oof_proba).reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if a < 0:
        a, b = 0.0, float(_logit(y.mean()))
    return {"a": a, "b": b, **edge_test(y, oof_proba)}


def apply_calibration(cal: dict, p_up: float) -> float:
    z = cal["a"] * float(_logit(p_up)) + cal["b"]
    return float(1 / (1 + np.exp(-z)))


def confidence_label(confidence: float, significant_edge: bool) -> str:
    """Unproven models are always LOW, however confident the raw score looks."""
    if not significant_edge or confidence < 0.55:
        return "LOW"
    return "HIGH" if confidence >= 0.62 else "MODERATE"


# ------------------------------------------------------------ train / predict
def train_final(dataset: pd.DataFrame) -> XGBClassifier:
    data = labeled(dataset)
    return new_model().fit(data[FEATURE_COLUMNS], data["target"].astype(int))


def _fmt(kind: str, v: float) -> str:
    if kind == "pct":
        return f"{v * 100:+.2f}%"
    if kind == "abs":
        return f"{v * 100:.2f}%"
    return f"{v:.2f}"


def predict_latest(model: XGBClassifier, dataset: pd.DataFrame, calibration: dict | None = None,
                   top_k: int = 5) -> dict:
    """Predict the direction after the newest row, with the top-k factors behind THIS prediction.

    Factors come from XGBoost's built-in per-prediction contributions (TreeSHAP, in log-odds):
    positive pushes toward UP, negative toward DOWN. They sum (with a bias term) to the model output.
    """
    row = dataset.iloc[[-1]]
    X = row[FEATURE_COLUMNS]
    p_up = float(model.predict_proba(X)[0, 1])
    direction = "UP" if p_up >= 0.5 else "DOWN"
    raw_conf = max(p_up, 1 - p_up)
    if calibration:  # probability that THIS call is right, learned from out-of-fold history
        p_cal = apply_calibration(calibration, p_up)
        confidence = p_cal if direction == "UP" else 1 - p_cal
        conf_label = confidence_label(confidence, calibration["significant_edge"])
    else:
        p_cal, confidence, conf_label = None, raw_conf, "UNCALIBRATED"

    contribs = model.get_booster().predict(DMatrix(X), pred_contribs=True)[0]
    log_odds = float(contribs.sum())  # includes bias as last element
    per_feature = contribs[:-1]
    order = np.argsort(-np.abs(per_feature))[:top_k]

    drivers = []
    for i in order:
        name = FEATURE_COLUMNS[i]
        label, kind = FEATURE_META[name]
        val = float(X.iloc[0][name])
        drivers.append({
            "feature": name, "label": label, "value": val, "value_text": _fmt(kind, val),
            "contribution": float(per_feature[i]),
            "pushes": "UP" if per_feature[i] > 0 else "DOWN",
        })

    parts = "; ".join(f"{d['label']} = {d['value_text']} (pushes {d['pushes']})" for d in drivers)
    reasoning = (f"AI simulation leans {direction} (calibrated confidence {confidence:.1%}, {conf_label}). "
                 f"Main factors: {parts}.")
    return {
        "as_of": row["date"].iloc[0].date(), "direction": direction, "confidence": confidence,
        "p_up": p_up, "p_up_calibrated": p_cal, "raw_confidence": raw_conf,
        "confidence_label": conf_label, "log_odds": log_odds, "drivers": drivers, "reasoning": reasoning,
    }


# ------------------------------------------------------------ persistence
def _model_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.replace('^', '_')}_{MODEL_VERSION}.json"


def save_model(model: XGBClassifier, ticker: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(ticker)
    model.save_model(path)
    return path


def load_model(ticker: str) -> XGBClassifier:
    model = new_model()
    model.load_model(_model_path(ticker))
    return model


def _calib_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.replace('^', '_')}_{MODEL_VERSION}_calib.json"


def save_calibration(cal: dict, ticker: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _calib_path(ticker)
    path.write_text(json.dumps(cal), encoding="utf-8")
    return path


def load_calibration(ticker: str) -> dict | None:
    path = _calib_path(ticker)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def store_prediction(session: Session, ticker: str, result: dict) -> int:
    """One prediction per (ticker, as-of date): update if it exists, else insert."""
    version = f"{MODEL_VERSION}-{datetime.now():%Y%m%d}"
    row = session.scalars(select(Prediction).where(
        Prediction.ticker == ticker, Prediction.pred_date == result["as_of"])).first()
    if row is None:
        row = Prediction(ticker=ticker, pred_date=result["as_of"])
        session.add(row)
    row.direction, row.confidence = result["direction"], result["confidence"]
    row.reasoning_text, row.model_version = result["reasoning"], version
    session.commit()
    return row.pred_id
