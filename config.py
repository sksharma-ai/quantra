"""Central config. Secrets come from environment (.env), never from source."""
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///quantra.db")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRY_HOURS = 24
BCRYPT_ROUNDS = 12

DISCLAIMER = (
    "QUANTRA provides AI-generated simulations and data analysis for educational "
    "purposes only. No financial advice is provided. Consult a SEBI-registered "
    "investment advisor before making investment decisions."
)

DEFAULT_TICKERS = {
    "RELIANCE.NS": ("Reliance Industries", "NSE", "Energy"),
    "TCS.NS": ("Tata Consultancy Services", "NSE", "IT"),
    "INFY.NS": ("Infosys", "NSE", "IT"),
    "HDFCBANK.NS": ("HDFC Bank", "NSE", "Banking"),
    "AAPL": ("Apple Inc.", "NASDAQ", "Technology"),
    "MSFT": ("Microsoft Corp.", "NASDAQ", "Technology"),
    # Index tickers used for the market-level crash risk score
    "^NSEI": ("NIFTY 50", "NSE", "Index"),
    "^GSPC": ("S&P 500", "NYSE", "Index"),
}

# NewsAPI search queries for market-wide sentiment (used by the crash risk module)
MARKET_NEWS_QUERIES = {
    "^NSEI": "Nifty OR Sensex OR \"Indian stock market\"",
    "^GSPC": "\"S&P 500\" OR \"Wall Street\" OR \"Federal Reserve\"",
}
