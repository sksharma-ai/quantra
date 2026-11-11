import pytest
from datetime import date
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import Base, User, Stock, PriceData, Portfolio, PaperTrade, RiskScore
from db.session import make_engine


@pytest.fixture
def s():
    eng = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as sess:
        sess.add(Stock(ticker="TCS.NS", company_name="TCS", exchange="NSE"))
        sess.commit()
        yield sess


def test_eleven_tables():
    assert len(Base.metadata.tables) == 11


def test_user_unique_username(s):
    s.add(User(username="a", email="a@x.com", password_hash="h")); s.commit()
    s.add(User(username="a", email="b@x.com", password_hash="h"))
    with pytest.raises(IntegrityError):
        s.commit()


def test_price_unique_per_day(s):
    row = dict(ticker="TCS.NS", date=date(2026, 1, 1), open=10, high=11, low=9, close=10, volume=5)
    s.add(PriceData(**row)); s.commit()
    s.add(PriceData(**row))
    with pytest.raises(IntegrityError):
        s.commit()


def test_price_check_high_ge_open(s):
    s.add(PriceData(ticker="TCS.NS", date=date(2026, 1, 2), open=10, high=9, low=8, close=9, volume=1))
    with pytest.raises(IntegrityError):
        s.commit()


def test_foreign_key_enforced(s):
    s.add(PriceData(ticker="NOPE", date=date(2026, 1, 3), open=1, high=2, low=1, close=1, volume=1))
    with pytest.raises(IntegrityError):
        s.commit()


def test_trade_type_check(s):
    u = User(username="u", email="u@x.com", password_hash="h"); s.add(u); s.commit()
    p = Portfolio(user_id=u.user_id, portfolio_name="p"); s.add(p); s.commit()
    s.add(PaperTrade(portfolio_id=p.portfolio_id, ticker="TCS.NS", trade_type="YOLO", quantity=1, price_at_trade=1))
    with pytest.raises(IntegrityError):
        s.commit()


def test_risk_score_range(s):
    s.add(RiskScore(ticker="TCS.NS", score_date=date(2026, 1, 1), crash_risk_score=140, risk_level="HIGH"))
    with pytest.raises(IntegrityError):
        s.commit()
