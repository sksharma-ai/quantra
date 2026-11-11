import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    """Synthetic random-walk OHLCV, 300 business days."""
    rng = np.random.default_rng(42)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=n),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n), "adj_close": close,
    })
