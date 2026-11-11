"""Train + validate the XGBoost model per ticker, calibrate confidence, store prediction.

Run:  python -m scripts.train        (after scripts.ingest)
Writes reports/model_metrics.json  <- real numbers for the project report.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import config
from db.session import get_session
from ml.features import build_dataset
from ml.model import (MODEL_VERSION, fit_calibrator, predict_latest, save_calibration, save_model,
                      store_prediction, train_final, walk_forward_validate)
from scripts.init_db import init_db
from services.data_ingestion import load_prices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=list(config.DEFAULT_TICKERS))
    args = ap.parse_args()

    init_db()
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "model": MODEL_VERSION, "tickers": {}}
    print(f"{'ticker':12s} {'folds':>5s} {'acc':>6s} {'base':>6s} {'auc':>6s} {'p-val':>6s} "
          f"{'edge':>5s}  next-day (calibrated confidence)")
    with get_session() as s:
        for t in args.tickers:
            ds = build_dataset(load_prices(s, t))
            try:
                wf = walk_forward_validate(ds)
            except ValueError as exc:
                print(f"{t:12s} skipped: {exc}")
                continue
            cal = fit_calibrator(wf.pop("oof_proba"), wf.pop("oof_target"))
            save_calibration(cal, t)
            model = train_final(ds)
            save_model(model, t)
            res = predict_latest(model, ds, cal)
            store_prediction(s, t, res)
            m = wf["mean"]
            auc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "  n/a"
            edge = "YES" if cal["significant_edge"] else "no"
            print(f"{t:12s} {len(wf['folds']):5d} {m['accuracy']:6.3f} {m['baseline_accuracy']:6.3f} "
                  f"{auc:>6s} {cal['p_value']:6.3f} {edge:>5s}  "
                  f"{res['direction']} {res['confidence']:.1%} {res['confidence_label']}")
            report["tickers"][t] = {
                "walk_forward": wf, "calibration": cal,
                "prediction": {**{k: res[k] for k in ("direction", "confidence", "confidence_label",
                                                      "p_up", "p_up_calibrated", "reasoning", "drivers")},
                               "as_of": str(res["as_of"])},
            }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved reports/model_metrics.json")


if __name__ == "__main__":
    main()
