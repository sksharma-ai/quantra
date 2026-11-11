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
}
