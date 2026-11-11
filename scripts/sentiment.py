"""Score all unscored headlines and print a per-ticker summary.

Run:  python -m scripts.sentiment      (after scripts.ingest)
First run downloads FinBERT (~440 MB). Needs internet; falls back to VADER if it cannot load.
"""
import json
import logging
from pathlib import Path

import config
from db.session import get_session
from ml.sentiment import get_scorer, score_headlines, ticker_summary
from scripts.init_db import init_db


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()
    scorer, name = get_scorer()
    print(f"scoring model: {name}")
    summary = {"model": name, "tickers": {}}
    with get_session() as s:
        print(f"newly scored headlines: {score_headlines(s, scorer, name)}")
        print(f"{'ticker':12s} {'n':>4s} {'mean':>7s} {'pos':>4s} {'neu':>4s} {'neg':>4s}")
        for t in config.DEFAULT_TICKERS:
            r = ticker_summary(s, t)
            summary["tickers"][t] = r
            mean = f"{r['mean_score']:+.3f}" if r["mean_score"] is not None else "   n/a"
            print(f"{t:12s} {r['n']:4d} {mean:>7s} {r['positive']:4d} {r['neutral']:4d} {r['negative']:4d}")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/sentiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved reports/sentiment_summary.json")


if __name__ == "__main__":
    main()
