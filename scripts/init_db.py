"""Create all tables and seed default stocks. Run: python -m scripts.init_db"""
from datetime import datetime

import config
from db.models import Base, Stock
from db.session import engine, get_session


def init_db():
    Base.metadata.create_all(engine)
    with get_session() as s:
        for ticker, (name, exch, sector) in config.DEFAULT_TICKERS.items():
            if not s.get(Stock, ticker):
                s.add(Stock(ticker=ticker, company_name=name, exchange=exch,
                            sector=sector, last_updated=datetime.now()))
        s.commit()
    print(f"DB ready at {config.DATABASE_URL}. Tables: {len(Base.metadata.tables)}")


if __name__ == "__main__":
    init_db()
