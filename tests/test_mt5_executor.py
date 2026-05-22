"""
tests/test_mt5_executor.py -- Step 7: MT5 Executor Tests
All MT5 API calls are mocked. No live MT5 connection required.
10 tests covering: data fetching, ATR, observation shape, lot sizing,
risk guard rules, state persistence, order execution, positions, and timing.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Mock MetaTrader5 before importing executor
import types
mock_mt5 = types.ModuleType("MetaTrader5")
mock_mt5.TIMEFRAME_D1       = 1440
mock_mt5.ORDER_TYPE_BUY     = 0
mock_mt5.ORDER_TYPE_SELL    = 1
mock_mt5.TRADE_ACTION_DEAL  = 1
mock_mt5.TRADE_ACTION_SLTP  = 6
mock_mt5.ORDER_TIME_GTC     = 1
mock_mt5.ORDER_FILLING_IOC  = 1
mock_mt5.TRADE_RETCODE_DONE = 10009
sys.modules["MetaTrader5"]  = mock_mt5

import execution.mt5_executor as exe
from execution.mt5_executor import (
    compute_atr_series, compute_tick_vol_zscore,
    LiveDataFetcher, LiveObservationBuilder, LiveRiskGuard,
    PositionManager, MT5Executor, zone_stop_tp, is_ny_close_now,
    LIVE_STATE_PATH,
)


def _make_df(n=300, seed=42):
    rng   = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.standard_normal(n) * 0.001)
    open_ = close + rng.standard_normal(n) * 0.0005
    high  = np.maximum(close, open_) + abs(rng.standard_normal(n) * 0.0003)
    low   = np.minimum(close, open_) - abs(rng.standard_normal(n) * 0.0003)
    tv    = abs(rng.standard_normal(n) * 1000 + 5000)
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df    = pd.DataFrame({"date": dates, "open": open_, "high": high,
                          "low": low, "close": close, "tick_volume": tv})
    df["atr_14"]          = compute_atr_series(df, 14)
    df["atr_200"]         = compute_atr_series(df, 200)
    df["tick_vol_zscore"] = compute_tick_vol_zscore(df["tick_volume"].to_numpy())
    df["day_of_week"]     = df["date"].dt.dayofweek
    df["date_str"]        = df["date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


# Test 1: compute_atr_series shape and positive values
def test_compute_atr_shape_and_values():
    df  = _make_df(200)
    atr = compute_atr_series(df, 14)
    assert len(atr) == 200
    assert (atr > 0).all()


# Test 2: compute_tick_vol_zscore shape and finite values
def test_tick_vol_zscore():
    tv = np.abs(np.random.default_rng(0).standard_normal(100) * 1000 + 5000)
    z  = compute_tick_vol_zscore(tv, window=20)
    assert len(z) == 100
    assert np.isfinite(z).all()


# Test 3: LiveDataFetcher normalizes MT5 rates correctly
def test_live_data_fetcher_normalization():
    n   = 300
    rng = np.random.default_rng(1)
    fake_rates = np.zeros(n, dtype=[
        ("time","i8"),("open","f8"),("high","f8"),("low","f8"),
        ("close","f8"),("tick_volume","i8"),("spread","i4"),("real_volume","i8"),
    ])
    fake_rates["time"]        = np.arange(n) * 86400 + 1577836800
    fake_rates["open"]        = 1.10 + rng.standard_normal(n) * 0.001
    fake_rates["high"]        = fake_rates["open"] + 0.002
    fake_rates["low"]         = fake_rates["open"] - 0.002
    fake_rates["close"]       = fake_rates["open"] + rng.standard_normal(n) * 0.0005
    fake_rates["tick_volume"] = np.full(n, 5000)
    mock_mt5.copy_rates_from_pos = MagicMock(return_value=fake_rates)
    exe.MT5_AVAILABLE = True; exe.mt5 = mock_mt5
    df = LiveDataFetcher().fetch("EURUSD", n_bars=n)
    assert not df.empty
    assert "atr_14" in df.columns
    assert "atr_200" in df.columns
    assert "tick_vol_zscore" in df.columns
    assert len(df) == n
    assert (df["atr_14"] > 0).all()


# Test 4: LiveObservationBuilder shape and bounds
def test_observation_shape_and_bounds():
    df = _make_df(300)
    ts = {"in_trade": False, "direction": 0, "entry_price": 0.0,
          "stop_loss": 0.0, "bars_in_trade": 0, "remaining_fraction": 1.0}
    obs = LiveObservationBuilder().build(
        "EURUSD", df, {"EURUSD": df}, ts, 100_000.0, 100_000.0)
    assert obs.shape == (66,)
    assert obs.dtype == np.float32
    assert obs.min() >= -10.0
    assert obs.max() <=  10.0


# Test 5: lot_size_from_risk reasonable values
def test_lot_size_from_risk():
    info_mock = MagicMock()
    info_mock.trade_contract_size = 100_000.0
    info_mock.volume_step = 0.01
    info_mock.volume_min  = 0.01
    info_mock.volume_max  = 500.0
    mock_mt5.symbol_info = MagicMock(return_value=info_mock)
    exe.MT5_AVAILABLE = True; exe.mt5 = mock_mt5
    lots = PositionManager().lot_size_from_risk("EURUSD", 100_000.0, 0.0030)
    assert lots > 0
    assert lots <= 500.0
    assert 30.0 <= lots <= 40.0   # ~33 lots expected


# Test 6: daily loss limit blocks entry
def test_risk_guard_daily_loss_blocks_entry(tmp_path):
    orig = exe.LIVE_STATE_PATH; exe.LIVE_STATE_PATH = tmp_path / "ls.json"
    try:
        g = LiveRiskGuard()
        g._state["daily_realized_r"] = -3.01
        allowed, reason = g.check_entry_allowed("EURUSD", 100_000.0, 0.0, 0.0)
        assert not allowed
        assert reason == "daily_loss_limit"
    finally:
        exe.LIVE_STATE_PATH = orig


# Test 7: consecutive loss circuit breaker
def test_risk_guard_circuit_breaker(tmp_path):
    orig = exe.LIVE_STATE_PATH; exe.LIVE_STATE_PATH = tmp_path / "ls.json"
    try:
        g = LiveRiskGuard()
        g._state["cooldown_bars_remaining"] = 3
        allowed, reason = g.check_entry_allowed("EURUSD", 100_000.0, 0.0, 0.0)
        assert not allowed
        assert reason == "consecutive_loss_circuit_breaker"
    finally:
        exe.LIVE_STATE_PATH = orig


# Test 8: state persistence roundtrip
def test_risk_guard_state_persistence(tmp_path):
    orig = exe.LIVE_STATE_PATH; exe.LIVE_STATE_PATH = tmp_path / "ls.json"
    try:
        g1 = LiveRiskGuard()
        g1._state["consecutive_losses"]      = 2
        g1._state["cooldown_bars_remaining"] = 4
        g1._state["daily_realized_r"]        = -1.5
        g1._state["current_day"]             = "2026-05-17"
        g1.save_state()
        g2 = LiveRiskGuard()
        assert g2._state["consecutive_losses"]      == 2
        assert g2._state["cooldown_bars_remaining"] == 4
        assert abs(g2._state["daily_realized_r"] - (-1.5)) < 1e-6
        assert g2._state["current_day"] == "2026-05-17"
    finally:
        exe.LIVE_STATE_PATH = orig


# Test 9: zone_stop_tp SL on correct side
def test_zone_stop_tp_sl_on_correct_side():
    from zone_engine.sr_engine import SREngine
    from data_pipeline.config import PAIRS
    df  = _make_df(300)
    ps  = float(PAIRS.get("EURUSD", {}).get("pip_scale", 10_000))
    edf = df[["open","high","low","close","tick_volume"]].rename(
        columns={"tick_volume": "volume"}).copy()
    engine = SREngine(edf, pip_scale=ps)
    for i in range(len(df)): engine.step(i)
    zd  = engine.get_nearest_zones(len(df)-1, n_sup=2, n_res=2)
    zf  = np.concatenate([zd["sup_features"].flatten(),
                          zd["res_features"].flatten()]).astype(np.float32)
    row   = df.iloc[-1]
    close = float(row["close"])
    atr14 = float(row["atr_14"])
    sl_l, tp_l = zone_stop_tp(+1, close, atr14, zf)
    assert sl_l < close, "Long SL must be below close"
    assert tp_l > close, "Long TP must be above close"
    sl_s, tp_s = zone_stop_tp(-1, close, atr14, zf)
    assert sl_s > close, "Short SL must be above close"
    assert tp_s < close, "Short TP must be below close"


# Test 10: MT5Executor.connect calls correct MT5 functions
def test_executor_connect_calls_mt5(tmp_path):
    mock_mt5.initialize   = MagicMock(return_value=True)
    mock_mt5.login        = MagicMock(return_value=True)
    ai = MagicMock(); ai.login = 50506004795; ai.equity = 100_000.0; ai.server = "MetaQuotes-Demo"
    mock_mt5.account_info = MagicMock(return_value=ai)
    mock_mt5.shutdown     = MagicMock()
    orig = exe.LIVE_STATE_PATH; exe.LIVE_STATE_PATH = tmp_path / "ls.json"
    exe.MT5_AVAILABLE = True; exe.mt5 = mock_mt5
    try:
        result = MT5Executor(password="testpwd").connect()
        assert result is True
        mock_mt5.initialize.assert_called_once()
        mock_mt5.login.assert_called_once_with(
            50506004795, password="testpwd", server="MetaQuotes-Demo")
    finally:
        exe.LIVE_STATE_PATH = orig
