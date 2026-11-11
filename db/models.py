"""SQLAlchemy models for all 11 QUANTRA tables (synopsis section 8.4 / 9)."""
from datetime import datetime, date

from sqlalchemy import (
    Boolean, CheckConstraint, Date, DateTime, Float, ForeignKey, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    data_erasure_requested: Mapped[bool] = mapped_column(Boolean, default=False)

    portfolios: Mapped[list["Portfolio"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (CheckConstraint("role IN ('user','admin')", name="ck_users_role"),)


class Stock(Base):
    __tablename__ = "stocks"
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    company_name: Mapped[str] = mapped_column(String(200), nullable=False)
    exchange: Mapped[str] = mapped_column(String(10), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(100))
    market_cap: Mapped[float | None] = mapped_column(Float)
    last_updated: Mapped[datetime | None] = mapped_column(DateTime)

    __table_args__ = (
        CheckConstraint("exchange IN ('NSE','BSE','NYSE','NASDAQ')", name="ck_stocks_exchange"),
    )


class PriceData(Base):
    __tablename__ = "price_data"
    price_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    adj_close: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),
        CheckConstraint("open > 0", name="ck_price_open"),
        CheckConstraint("high >= open", name="ck_price_high"),
        CheckConstraint("low <= open", name="ck_price_low"),
        CheckConstraint("close > 0", name="ck_price_close"),
        CheckConstraint("volume >= 0", name="ck_price_volume"),
    )


class TechnicalIndicator(Base):
    __tablename__ = "technical_indicators"
    indicator_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    rsi: Mapped[float | None] = mapped_column(Float)
    macd: Mapped[float | None] = mapped_column(Float)
    macd_signal: Mapped[float | None] = mapped_column(Float)
    bb_upper: Mapped[float | None] = mapped_column(Float)
    bb_lower: Mapped[float | None] = mapped_column(Float)
    sma_21: Mapped[float | None] = mapped_column(Float)
    ema_9: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ind_ticker_date"),)


class Prediction(Base):
    __tablename__ = "predictions"
    pred_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False)
    pred_date: Mapped[date] = mapped_column(Date, nullable=False)
    direction: Mapped[str] = mapped_column(String(4), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning_text: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    __table_args__ = (
        CheckConstraint("direction IN ('UP','DOWN')", name="ck_pred_direction"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_pred_conf"),
    )


class NewsHeadline(Base):
    __tablename__ = "news_headlines"
    news_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str | None] = mapped_column(ForeignKey("stocks.ticker"))
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(100))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    url: Mapped[str | None] = mapped_column(String(500))

    sentiment: Mapped["SentimentScore | None"] = relationship(back_populates="news", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("url", name="uq_news_url"),)


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"
    sent_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_id: Mapped[int] = mapped_column(ForeignKey("news_headlines.news_id"), nullable=False, unique=True)
    sentiment_label: Mapped[str] = mapped_column(String(10), nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    model_used: Mapped[str | None] = mapped_column(String(50))
    scored_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    news: Mapped[NewsHeadline] = relationship(back_populates="sentiment")

    __table_args__ = (
        CheckConstraint("sentiment_label IN ('Positive','Neutral','Negative')", name="ck_sent_label"),
    )


class RiskScore(Base):
    __tablename__ = "risk_scores"
    risk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False)
    score_date: Mapped[date] = mapped_column(Date, nullable=False)
    crash_risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    volatility_component: Mapped[float | None] = mapped_column(Float)
    sentiment_component: Mapped[float | None] = mapped_column(Float)
    anomaly_component: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(4), nullable=False)

    __table_args__ = (
        UniqueConstraint("ticker", "score_date", name="uq_risk_ticker_date"),
        CheckConstraint("crash_risk_score >= 0 AND crash_risk_score <= 100", name="ck_risk_range"),
        CheckConstraint("risk_level IN ('LOW','MED','HIGH')", name="ck_risk_level"),
    )


class Portfolio(Base):
    __tablename__ = "portfolios"
    portfolio_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    portfolio_name: Mapped[str] = mapped_column(String(100), nullable=False)
    virtual_cash: Mapped[float] = mapped_column(Float, nullable=False, default=1_000_000.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    last_updated: Mapped[datetime | None] = mapped_column(DateTime)

    user: Mapped[User] = relationship(back_populates="portfolios")
    trades: Mapped[list["PaperTrade"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")

    __table_args__ = (CheckConstraint("virtual_cash >= 0", name="ck_portfolio_cash"),)


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    trade_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.portfolio_id"), nullable=False)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False)
    trade_type: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price_at_trade: Mapped[float] = mapped_column(Float, nullable=False)
    trade_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())
    pnl: Mapped[float | None] = mapped_column(Float)
    trigger_reason: Mapped[str | None] = mapped_column(Text)

    portfolio: Mapped[Portfolio] = relationship(back_populates="trades")

    __table_args__ = (
        CheckConstraint("trade_type IN ('BUY','SELL','HOLD')", name="ck_trade_type"),
        CheckConstraint("quantity > 0", name="ck_trade_qty"),
        CheckConstraint("price_at_trade > 0", name="ck_trade_price"),
    )


class MacroEvent(Base):
    __tablename__ = "macro_events"
    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_name: Mapped[str] = mapped_column(String(200), nullable=False)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    event_type: Mapped[str] = mapped_column(String(10), nullable=False)
    expected_impact: Mapped[str] = mapped_column(String(4), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("event_type IN ('RBI','FED','BUDGET','EARNINGS')", name="ck_event_type"),
        CheckConstraint("expected_impact IN ('HIGH','MED','LOW')", name="ck_event_impact"),
    )
