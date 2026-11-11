import numpy as np
import pandas as pd
import pytest


def make_prices(n: int, seed: int = 42, rho: float = 0.0, vol: float = 0.015) -> pd.DataFrame:
    """Synthetic OHLCV. rho=0 -> random walk; rho=-0.9 -> returns strongly mean-revert (learnable)."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for t in range(1, n):
        ret[t] = rho * ret[t - 1] + rng.normal(0.0003, vol)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=n),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n), "adj_close": close,
    })


@pytest.fixture
def prices():
    return make_prices(300)


@pytest.fixture
def long_prices():
    return make_prices(700)
