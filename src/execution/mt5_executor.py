"""
Step 7B: MT5 Live Executor — Nick Shawn Forex Trading Bot
=========================================================
Connects to MetaTrader 5, fetches live daily candles for all 24 pairs,
builds the identical 66-dim observation vector used during training,
runs the trained PPO model deterministically, and executes orders.

Enforces all 5 RiskGuardWrapper rules in live mode.
Persists risk state between daily runs to models/live_state.json.

Account: MetaQuotes-Demo, login 50506004795.
Password: never hardcoded -- pass via --password CLI arg or MT5_PASSWORD env var.

Usage:
  python src/execution/mt5_executor.py --run-now  --password <your_password>
  python src/execution/mt5_executor.py --schedule --password <your_password>
  python src/execution/mt5_executor.py --status   --password <your_password>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

from stable_baselines3 import PPO

from data_pipeline.config import PAIRS, SPREADS
from zone_engine.sr_engine import SREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FINAL_MODEL_PATH = ROOT / "models" / "final" / "final_model.zip"
BEST_MODEL_PATH  = ROOT / "models" / "final" / "best" / "best_model.zip"
LIVE_STATE_PATH  = ROOT / "models" / "live_state.json"

MT5_LOGIN      = 50506004795
MT5_SERVER     = "MetaQuotes-Demo"
MT5_MAGIC      = 234567
INITIAL_EQUITY = 100_000.0

RISK_PER_TRADE  = 0.10
HISTORY_BARS    = 300
CORR_WINDOW     = 20
RISK_WINDOW     = 10

PASS          = 0
ENTER_LONG    = 1
ENTER_SHORT   = 2
HOLD          = 3
CLOSE         = 4
PARTIAL_CLOSE = 5

DAILY_LOSS_LIMIT     = -3.0
CONSECUTIVE_LOSS_MAX = 3
COOLDOWN_BARS        = 5
EQUITY_CURVE_THRESH  = 0.85
MAX_BARS_IN_TRADE    = 10

_USD_PAIRS = frozenset({
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"
})
_JPY_PAIRS = frozenset({
    "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"
})


# ===========================================================================
#  ATR COMPUTATION (matches clean.py Wilder EWM exactly)
# ===========================================================================

def compute_atr_series(df: pd.DataFrame, period: int) -> np.ndarray:
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)

    prev_cl     = np.empty_like(cl)
    prev_cl[0]  = cl[0]
    prev_cl[1:] = cl[:-1]

    tr = np.maximum(
        hi - lo,
        np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)),
    )

    n, alpha = len(tr), 1.0 / period
    atr = np.empty(n, dtype=float)
    for i in range(min(period, n)):
        atr[i] = tr[: i + 1].mean()
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def compute_tick_vol_zscore(tick_volume: np.ndarray, window: int = 20) -> np.ndarray:
    n, z = len(tick_volume), np.zeros(len(tick_volume), dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        w = tick_volume[start: i + 1]
        if len(w) < 2:
            z[i] = 0.0
            continue
        std = w.std()
        z[i] = (tick_volume[i] - w.mean()) / (std + 1e-8) if std > 0 else 0.0
    return z


# ===========================================================================
#  LIVE DATA FETCHER
# ===========================================================================

class LiveDataFetcher:
    def fetch(self, symbol: str, n_bars: int = HISTORY_BARS) -> pd.DataFrame:
        if not MT5_AVAILABLE or mt5 is None:
            raise RuntimeError("MetaTrader5 package not available.")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, n_bars)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["date", "open", "high", "low", "close", "tick_volume"]].copy()
        df["atr_14"]          = compute_atr_series(df, 14)
        df["atr_200"]         = compute_atr_series(df, 200)
        df["tick_vol_zscore"] = compute_tick_vol_zscore(df["tick_volume"].to_numpy())
        df["day_of_week"]     = df["date"].dt.dayofweek
        df["date_str"]        = df["date"].dt.strftime("%Y-%m-%d")
        return df.reset_index(drop=True)


# ===========================================================================
#  CORRELATION BUILDER
# ===========================================================================

class CorrelationBuilder:
    _USD_PS = [("EURUSD",-1),("GBPUSD",-1),("AUDUSD",-1),("NZDUSD",-1),
               ("USDJPY",+1),("USDCHF",+1),("USDCAD",+1)]
    _JPY_PS = [("USDJPY",-1),("EURJPY",-1),("GBPJPY",-1),("AUDJPY",-1),
               ("CADJPY",-1),("CHFJPY",-1),("NZDJPY",-1)]
    _GBP_PS = [("GBPUSD",+1),("EURGBP",-1),("GBPJPY",+1),
               ("GBPAUD",+1),("GBPCAD",+1),("GBPNZD",+1)]

    def build(self, pair_data: Dict[str, pd.DataFrame], date_str: str
              ) -> Tuple[float, float, float, float]:
        ret_z: Dict[str, float] = {}
        for pair, df in pair_data.items():
            if df.empty or len(df) < CORR_WINDOW + 1:
                continue
            log_ret = np.log(df["close"] / df["close"].shift(1))
            rm  = log_ret.rolling(CORR_WINDOW).mean()
            rs  = log_ret.rolling(CORR_WINDOW).std().replace(0.0, 1e-8)
            z   = (log_ret - rm) / rs
            last = z.iloc[-1]
            ret_z[pair] = float(last) if not np.isnan(last) else 0.0

        def _idx(ps):
            vals = [ret_z.get(p, np.nan)*s for p,s in ps if p in ret_z]
            valid = [v for v in vals if not np.isnan(v)]
            return float(np.mean(valid)) if valid else 0.0

        usd = _idx(self._USD_PS)
        jpy = _idx(self._JPY_PS)
        gbp = _idx(self._GBP_PS)
        risk = 0.0
        if "AUDJPY" in pair_data and len(pair_data["AUDJPY"]) > RISK_WINDOW:
            r = pair_data["AUDJPY"]["close"].pct_change(RISK_WINDOW).iloc[-1]
            risk = 1.0 if (not np.isnan(r) and r > 0) else -1.0
        return usd, jpy, gbp, risk


# ===========================================================================
#  LIVE OBSERVATION BUILDER  (matches ForexTradingEnv._get_observation exactly)
# ===========================================================================

class LiveObservationBuilder:
    def __init__(self):
        self._corr = CorrelationBuilder()

    def build(self, pair, df, pair_data_all, trade_state,
              account_equity, account_peak_equity) -> np.ndarray:
        if df.empty or len(df) < 10:
            return np.zeros(66, dtype=np.float32)

        row    = df.iloc[-1]
        close  = max(float(row["close"]), 1e-8)
        atr14  = max(float(row["atr_14"]), 1e-8)
        atr200 = max(float(row["atr_200"]), 1e-8)
        tick_z = float(row["tick_vol_zscore"])
        dow    = float(row["day_of_week"]) / 4.0

        # Zone features [48]
        pip_scale = float(PAIRS.get(pair, {}).get("pip_scale", 10_000))
        engine_df = df[["open","high","low","close","tick_volume"]].rename(
            columns={"tick_volume": "volume"}).copy()
        engine = SREngine(engine_df, pip_scale=pip_scale)
        last_i = len(df) - 1
        for i in range(last_i + 1):
            engine.step(i)
        zd  = engine.get_nearest_zones(last_i, n_sup=2, n_res=2)
        zf  = np.concatenate([
            zd["sup_features"].flatten(),
            zd["res_features"].flatten(),
        ]).astype(np.float32)

        # Price context [8]
        price_ctx = np.array([
            float(zf[1]), float(zf[13]), float(zf[25]), float(zf[37]),
            atr14 / close, atr200 / close, tick_z, dow,
        ], dtype=np.float32)

        # Trade state [6]
        in_trade  = trade_state.get("in_trade", False)
        direction = trade_state.get("direction", 0)
        entry_px  = trade_state.get("entry_price", 0.0)
        sl_px     = trade_state.get("stop_loss", 0.0)
        bars_in   = trade_state.get("bars_in_trade", 0)
        rem_frac  = trade_state.get("remaining_fraction", 1.0)

        ur = 0.0
        if in_trade and abs(entry_px - sl_px) > 1e-8:
            ur = float(np.clip(
                (close - entry_px) * direction / abs(entry_px - sl_px) * rem_frac,
                -3.0, 3.0))

        trade_vec = np.array([
            1.0 if in_trade else 0.0,
            (direction + 1) / 2.0,
            ur,
            min(bars_in / 20.0, 5.0),
            account_equity / max(INITIAL_EQUITY, 1e-8),
            float(np.clip(1.0 - account_equity / max(account_peak_equity, 1e-8), 0, 1)),
        ], dtype=np.float32)

        # Correlation [4]
        usd, jpy, gbp, risk = self._corr.build(pair_data_all, str(row["date_str"]))
        corr = np.array([
            float(np.tanh(usd)), float(np.tanh(jpy)),
            float(np.tanh(gbp)), float(np.clip(risk, -1.0, 1.0)),
        ], dtype=np.float32)

        obs = np.concatenate([zf, price_ctx, trade_vec, corr])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)


# ===========================================================================
#  POSITION MANAGER
# ===========================================================================

class PositionManager:
    def get_open_position(self, symbol: str):
        if not MT5_AVAILABLE: return None
        positions = mt5.positions_get(symbol=symbol)
        return positions[0] if positions else None

    def lot_size_from_risk(self, symbol, equity, stop_distance_price) -> float:
        if not MT5_AVAILABLE or stop_distance_price < 1e-8: return 0.01
        info = mt5.symbol_info(symbol)
        if info is None: return 0.01
        risk_amount = equity * RISK_PER_TRADE
        raw = risk_amount / (stop_distance_price * float(info.trade_contract_size))
        step = float(info.volume_step)
        lots = max(float(info.volume_min),
                   min(float(info.volume_max), round(raw / step) * step))
        return round(lots, 2)

    def open_position(self, symbol, direction, entry_price,
                      stop_loss, take_profit, lot_size, comment="NickShawn_RL") -> bool:
        if not MT5_AVAILABLE: return False
        tick  = mt5.symbol_info_tick(symbol)
        price = tick.ask if direction == 1 else tick.bid
        otype = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        req   = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
            "volume": float(lot_size), "type": otype, "price": float(price),
            "sl": float(stop_loss), "tp": float(take_profit),
            "deviation": 20, "magic": MT5_MAGIC, "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        d  = "BUY" if direction == 1 else "SELL"
        if ok: logger.info(f"  [{symbol}] {d} {lot_size:.2f} @ {price:.5f} SL={stop_loss:.5f}")
        else:  logger.error(f"  [{symbol}] Order FAILED retcode={getattr(result,'retcode','None')}")
        return ok

    def close_position(self, position) -> bool:
        if not MT5_AVAILABLE or position is None: return False
        direction = 1 if position.type == 1 else -1
        otype = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        tick  = mt5.symbol_info_tick(position.symbol)
        price = tick.ask if direction == 1 else tick.bid
        req   = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": position.symbol,
            "volume": float(position.volume), "type": otype,
            "position": position.ticket, "price": float(price),
            "deviation": 20, "magic": MT5_MAGIC, "comment": "NickShawn_RL_close",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok: logger.info(f"  [{position.symbol}] CLOSED @ {price:.5f}")
        else:  logger.error(f"  [{position.symbol}] Close FAILED")
        return ok

    def partial_close_position(self, position, fraction: float = 0.5) -> bool:
        if not MT5_AVAILABLE or position is None: return False
        vol  = max(round(position.volume * fraction, 2),
                   mt5.symbol_info(position.symbol).volume_min)
        d    = 1 if position.type == 1 else -1
        otype = mt5.ORDER_TYPE_BUY if d == 1 else mt5.ORDER_TYPE_SELL
        tick  = mt5.symbol_info_tick(position.symbol)
        price = tick.ask if d == 1 else tick.bid
        req   = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": position.symbol,
            "volume": float(vol), "type": otype, "position": position.ticket,
            "price": float(price), "deviation": 20, "magic": MT5_MAGIC,
            "comment": "NickShawn_RL_partial",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE: return False
        be_req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": position.symbol,
                  "position": position.ticket, "sl": float(position.price_open),
                  "tp": float(position.tp)}
        mt5.order_send(be_req)
        logger.info(f"  [{position.symbol}] PARTIAL {vol:.2f} lots, SL -> breakeven")
        return True


# ===========================================================================
#  LIVE RISK GUARD
# ===========================================================================

class LiveRiskGuard:
    def __init__(self):
        self._state = {"consecutive_losses": 0, "cooldown_bars_remaining": 0,
                       "daily_realized_r": 0.0, "current_day": None,
                       "equity_history": []}
        self._load_state()

    def _load_state(self):
        if LIVE_STATE_PATH.exists():
            try:
                with open(LIVE_STATE_PATH) as f:
                    self._state.update(json.load(f))
                logger.info(f"Risk guard state loaded.")
            except Exception as e:
                logger.warning(f"Could not load live state: {e}")

    def save_state(self):
        LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LIVE_STATE_PATH, "w") as f:
            json.dump(self._state, f, indent=2)

    def update_day(self, date_str: str):
        if date_str != self._state["current_day"]:
            self._state["current_day"]      = date_str
            self._state["daily_realized_r"] = 0.0

    def record_equity(self, equity: float):
        h = self._state["equity_history"]
        h.append(equity)
        self._state["equity_history"] = h[-20:]

    def record_trade_result(self, realized_r: float):
        self._state["daily_realized_r"] += realized_r
        if realized_r < 0.0:
            self._state["consecutive_losses"] += 1
            if self._state["consecutive_losses"] >= CONSECUTIVE_LOSS_MAX:
                self._state["cooldown_bars_remaining"] = COOLDOWN_BARS
                logger.warning("Circuit breaker: 3 losses -> 5-bar cooldown")
        else:
            self._state["consecutive_losses"] = 0
        if self._state["cooldown_bars_remaining"] > 0:
            self._state["cooldown_bars_remaining"] -= 1

    def check_entry_allowed(self, pair, equity, usd_z, jpy_z) -> Tuple[bool, str]:
        if self._state["daily_realized_r"] <= DAILY_LOSS_LIMIT:
            return False, "daily_loss_limit"
        if self._state["cooldown_bars_remaining"] > 0:
            return False, "consecutive_loss_circuit_breaker"
        if abs(np.tanh(usd_z)) > 0.7 and pair.upper() in _USD_PAIRS:
            return False, "correlation_guard_USD"
        if abs(np.tanh(jpy_z)) > 0.7 and pair.upper() in _JPY_PAIRS:
            return False, "correlation_guard_JPY"
        hist = self._state["equity_history"]
        if len(hist) >= 20:
            avg = float(np.mean(hist[-20:]))
            if avg > 0 and equity < avg * EQUITY_CURVE_THRESH:
                return False, "equity_curve_filter"
        return True, "none"


# ===========================================================================
#  ZONE STOP/TP  (matches environment._zone_stop_tp exactly)
# ===========================================================================

def zone_stop_tp(direction, close, atr14, zf) -> Tuple[float, float]:
    feat      = zf[0:12] if direction == 1 else zf[24:36]
    midpoint  = close - float(feat[1]) * atr14
    near_edge = close - float(feat[2]) * atr14
    far_edge  = 2.0 * midpoint - near_edge
    if direction == 1:
        sl = far_edge - 0.1 * atr14
        tp = close + abs(close - sl)
    else:
        sl = far_edge + 0.1 * atr14
        tp = close - abs(sl - close)
    if direction == 1 and sl >= close:
        sl = close - 1.5 * atr14; tp = close + abs(close - sl)
    if direction == -1 and sl <= close:
        sl = close + 1.5 * atr14; tp = close - abs(sl - close)
    return sl, tp


# ===========================================================================
#  NY CLOSE SCHEDULER HELPER
# ===========================================================================

def is_ny_close_now(tolerance_minutes: int = 10) -> bool:
    try:
        import pytz
        ny   = pytz.timezone("America/New_York")
        now  = datetime.now(ny)
        if now.weekday() >= 5: return False
        close = now.replace(hour=17, minute=0, second=0, microsecond=0)
        return abs((now - close).total_seconds() / 60.0) <= tolerance_minutes
    except Exception:
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5: return False
        return now.hour == 22 and now.minute < tolerance_minutes


# ===========================================================================
#  MAIN EXECUTOR
# ===========================================================================

class MT5Executor:
    def __init__(self, password: str):
        self.password    = password
        self.fetcher     = LiveDataFetcher()
        self.obs_builder = LiveObservationBuilder()
        self.pos_manager = PositionManager()
        self.risk_guard  = LiveRiskGuard()
        self.model       = None
        self._peak_equity = INITIAL_EQUITY

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 not installed. Run: pip install MetaTrader5")
            return False
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False
        if not mt5.login(MT5_LOGIN, password=self.password, server=MT5_SERVER):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown(); return False
        info = mt5.account_info()
        logger.info(f"Connected: account {info.login} equity={info.equity:,.2f}")
        return True

    def disconnect(self):
        if MT5_AVAILABLE and mt5: mt5.shutdown(); logger.info("MT5 disconnected.")

    def load_model(self) -> bool:
        path = FINAL_MODEL_PATH if FINAL_MODEL_PATH.exists() else BEST_MODEL_PATH
        if not path.exists():
            logger.error("No trained model. Run: python src/rl_agent/final_trainer.py")
            return False
        self.model = PPO.load(str(path), device="cpu")
        logger.info(f"Model loaded from {path.name}")
        return True

    def run_daily(self):
        account = mt5.account_info()
        if account is None: logger.error("Cannot get account info."); return
        equity = float(account.equity)
        self._peak_equity = max(self._peak_equity, equity)
        today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.risk_guard.update_day(today)
        self.risk_guard.record_equity(equity)
        logger.info(f"Daily run | date={today} | equity=${equity:,.2f}")

        all_pairs  = list(PAIRS.keys())
        pair_data  = {}
        for pair in all_pairs:
            df = self.fetcher.fetch(pair)
            if not df.empty: pair_data[pair] = df

        if not pair_data: logger.error("No pair data — aborting."); return

        for pair in all_pairs:
            if pair not in pair_data: continue
            df       = pair_data[pair]
            position = self.pos_manager.get_open_position(pair)
            row      = df.iloc[-1]
            close    = float(row["close"])
            atr14    = max(float(row["atr_14"]), 1e-8)

            in_trade  = position is not None
            direction = (1 if position.type == 0 else -1) if in_trade else 0
            bars_in   = max(1, int((time.time() - position.time) / 86400)) if in_trade else 0

            trade_state = {
                "in_trade": in_trade, "direction": direction,
                "entry_price": float(position.price_open) if in_trade else 0.0,
                "stop_loss":   float(position.sl)         if in_trade else 0.0,
                "bars_in_trade": bars_in, "remaining_fraction": 1.0,
            }

            # Rule 4: max bars force close
            if in_trade and bars_in > MAX_BARS_IN_TRADE:
                closed = self.pos_manager.close_position(position)
                if closed:
                    risk = abs(trade_state["entry_price"] - trade_state["stop_loss"])
                    r    = (close - trade_state["entry_price"]) * direction / max(risk, 1e-8)
                    self.risk_guard.record_trade_result(r)
                continue

            obs = self.obs_builder.build(
                pair, df, pair_data, trade_state, equity, self._peak_equity)

            action_arr, _ = self.model.predict(obs, deterministic=True)
            action = int(action_arr)

            usd_z = float(np.arctanh(np.clip(obs[62], -0.9999, 0.9999)))
            jpy_z = float(np.arctanh(np.clip(obs[63], -0.9999, 0.9999)))

            if action in (ENTER_LONG, ENTER_SHORT):
                allowed, reason = self.risk_guard.check_entry_allowed(
                    pair, equity, usd_z, jpy_z)
                if not allowed:
                    logger.info(f"  [{pair}] Entry blocked: {reason}")
                    action = PASS

            actions = {PASS:"PASS",ENTER_LONG:"ENTER_LONG",ENTER_SHORT:"ENTER_SHORT",
                       HOLD:"HOLD",CLOSE:"CLOSE",PARTIAL_CLOSE:"PARTIAL_CLOSE"}
            logger.info(f"  [{pair}] Action: {actions.get(action, 'UNKNOWN')}")

            if action in (ENTER_LONG, ENTER_SHORT) and not in_trade:
                d  = 1 if action == ENTER_LONG else -1
                sl, tp = zone_stop_tp(d, close, atr14, obs[:48])
                stop_d = abs(close - sl)
                if stop_d > 1e-8:
                    lots = self.pos_manager.lot_size_from_risk(pair, equity, stop_d)
                    self.pos_manager.open_position(pair, d, close, sl, tp, lots)

            elif action == CLOSE and in_trade:
                closed = self.pos_manager.close_position(position)
                if closed:
                    risk = abs(trade_state["entry_price"] - trade_state["stop_loss"])
                    r    = (close - trade_state["entry_price"]) * direction / max(risk, 1e-8)
                    self.risk_guard.record_trade_result(r)

            elif action == PARTIAL_CLOSE and in_trade:
                self.pos_manager.partial_close_position(position)

        self.risk_guard.save_state()
        logger.info("Daily run complete.")

    def print_status(self):
        account   = mt5.account_info()
        positions = mt5.positions_get() or []
        W = 62
        print("=" * W)
        print("  MT5 EXECUTOR STATUS")
        print("-" * W)
        print(f"  Account  : {account.login}")
        print(f"  Balance  : ${account.balance:>12,.2f}")
        print(f"  Equity   : ${account.equity:>12,.2f}")
        if positions:
            print(f"\n  Open positions ({len(positions)}):")
            for pos in positions:
                d = "LONG" if pos.type == 0 else "SHORT"
                print(f"  {pos.symbol:<12} {d:<6} {pos.volume:.2f} lots  P&L={pos.profit:.2f}")
        else:
            print("  No open positions.")
        state = self.risk_guard._state
        print(f"\n  Daily realized R   : {state['daily_realized_r']:+.4f}")
        print(f"  Consecutive losses : {state['consecutive_losses']}")
        print(f"  Cooldown remaining : {state['cooldown_bars_remaining']} bars")
        print("=" * W)

    def run_scheduler(self):
        logger.info("Scheduler started. Waiting for NY close (17:00 EST)...")
        logger.info("Press Ctrl+C to stop.")
        fired_today = None
        try:
            while True:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if is_ny_close_now() and fired_today != today:
                    logger.info("NY close detected — running daily analysis...")
                    self.run_daily()
                    fired_today = today
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")


# ===========================================================================
#  CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Step 7B: MT5 Live Executor",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--password", type=str, default=os.environ.get("MT5_PASSWORD", ""),
                   help="MT5 password (or set MT5_PASSWORD env var)")
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--run-now",  action="store_true")
    m.add_argument("--schedule", action="store_true")
    m.add_argument("--status",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.password:
        print("ERROR: MT5 password required.")
        print("  Use --password <pwd>  OR  set MT5_PASSWORD environment variable.")
        raise SystemExit(1)
    executor = MT5Executor(password=args.password)
    if not executor.connect(): raise SystemExit(1)
    try:
        if args.status:
            executor.print_status(); return
        if not executor.load_model(): raise SystemExit(1)
        if args.run_now:   executor.run_daily()
        elif args.schedule: executor.run_scheduler()
    finally:
        executor.disconnect()


if __name__ == "__main__":
    main()
