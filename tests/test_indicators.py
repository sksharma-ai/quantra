import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, build_dataset, engineer_features
from ml.indicators import add_indicators, calc_bollinger, calc_rsi


def test_rsi_bounded(prices):
    r = calc_rsi(prices["close"]).dropna()
    assert len(r) > 200 and r.between(0, 100).all()


def test_rsi_all_up_is_100():
    assert calc_rsi(pd.Series(np.arange(1.0, 60.0))).iloc[-1] == 100.0


def test_bollinger_order(prices):
    bb = calc_bollinger(prices["close"]).dropna()
    assert (bb["bb_upper"] > bb["bb_mid"]).all() and (bb["bb_mid"] > bb["bb_lower"]).all()


def test_all_indicator_columns(prices):
    out = add_indicators(prices)
    for col in ["rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
                "sma_7", "sma_21", "sma_50", "ema_9", "ema_21", "volume_change"]:
        assert col in out.columns


def test_dataset_shape_and_nans(prices):
    ds = build_dataset(prices)
    assert not ds[FEATURE_COLUMNS].isna().any().any()
    assert np.isinf(ds[FEATURE_COLUMNS].to_numpy()).sum() == 0
    assert ds["target"].iloc[:-1].isin([0.0, 1.0]).all()
    assert pd.isna(ds["target"].iloc[-1])  # latest day has no known next-day label


def test_target_matches_next_day(prices):
    f = engineer_features(prices)
    expected = (prices["close"].shift(-1) > prices["close"]).astype(float)[:-1]
    assert (f["target"][:-1] == expected).all()


def test_no_lookahead(prices):
    """Changing future prices must not change features of earlier rows."""
    cut = 200
    a = engineer_features(prices)
    altered = prices.copy()
    altered.loc[cut:, ["open", "high", "low", "close"]] *= 3
    b = engineer_features(altered)
    pd.testing.assert_frame_equal(a.loc[:cut - 1, FEATURE_COLUMNS], b.loc[:cut - 1, FEATURE_COLUMNS])
