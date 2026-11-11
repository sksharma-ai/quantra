"""Compute crash risk for every tracked ticker, store it, and check it against what happened next.

Run:  python -m scripts.risk       (after scripts.ingest and scripts.sentiment)
Writes reports/risk_summary.json  <- real numbers for the project report.
"""
import json
from datetime import datetime
from pathlib import Path

import config
from db.session import get_session
from ml.risk import compute_risk, evaluate_risk, risk_history, store_risk_history, store_risk_score
from ml.sentiment import aggregate_daily_sentiment
from scripts.init_db import init_db
from services.data_ingestion import load_prices


def main():
    init_db()
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "tickers": {}}
    print(f"{'ticker':12s} {'score':>6s} {'level':>5s} {'vol':>5s} {'anom':>5s} {'sent':>5s} {'news':>4s}"
          f" | {'rho':>6s} {'HIGH n':>6s} {'lift':>5s}")
    with get_session() as s:
        for t in config.DEFAULT_TICKERS:
            prices = load_prices(s, t)
            try:
                res = compute_risk(prices, aggregate_daily_sentiment(s, t))
            except ValueError as exc:
                print(f"{t:12s} skipped: {exc}")
                continue
            hist = risk_history(prices)
            store_risk_history(s, t, hist)
            store_risk_score(s, t, res)           # latest row, with sentiment, overwrites same date
            ev = evaluate_risk(prices, hist)
            f = lambda x: f"{x:5.0f}" if x is not None else "  n/a"
            lift = f"{ev['lift_high_vs_base']:5.2f}" if ev["lift_high_vs_base"] is not None else "  n/a"
            print(f"{t:12s} {res['score']:6.1f} {res['level']:>5s} {f(res['volatility_component'])} "
                  f"{f(res['anomaly_component'])} {f(res['sentiment_component'])} {res['n_headlines']:4d}"
                  f" | {ev['spearman_rho']:+6.3f} {ev['by_level']['HIGH']['n']:6d} {lift}")
            report["tickers"][t] = {"latest": {**res, "as_of": str(res["as_of"])}, "evaluation": ev}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/risk_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved reports/risk_summary.json")
    print("rho = rank correlation of risk score vs next-10-day worst drop (positive is good).")
    print("lift = how much likelier a >5% drop is after a HIGH day than on any day (>1 is good).")


if __name__ == "__main__":
    main()
