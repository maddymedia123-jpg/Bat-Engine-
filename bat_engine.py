from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from market_data import compute_atr, compute_rsi, compute_stoch_rsi


@dataclass
class BatScanResult:
    symbol: str
    coin: str
    timeframe: str
    state: int  # 0: idle, 1: READY TO BE, 2: CONFIRMED, 3: invalidated
    status_label: str  # "READY TO BE" | "CONFIRMED" | "IDLE"
    sig_code: str  # e.g. R2, W1, C3, "-"
    pattern_type: str  # "BULLISH" | "BEARISH" | ""
    overstretched: bool
    conf_score: int  # 0..3
    conf_pct: float  # e.g. 33.3, 66.7, 100.0
    bad_confluences: List[str] = field(default_factory=list)
    entry_d: Optional[float] = None
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    zone_low: Optional[float] = None
    zone_high: Optional[float] = None
    zone_label: str = "-"
    beyond_optimal_entry: bool = False
    optimal_entry_status: str = "-"
    last_close: Optional[float] = None
    pivots: Dict[str, Tuple[int, float]] = field(default_factory=dict)
    leg_statuses: Dict[str, str] = field(default_factory=dict)


def _pivot_highs(high: np.ndarray, left: int, right: int) -> List[int]:
    idxs: List[int] = []
    n = len(high)
    for i in range(left, n - right):
        val = high[i]
        if np.all(high[i - left : i] < val) and np.all(high[i + 1 : i + right + 1] <= val):
            idxs.append(i)
    return idxs


def _pivot_lows(low: np.ndarray, left: int, right: int) -> List[int]:
    idxs: List[int] = []
    n = len(low)
    for i in range(left, n - right):
        val = low[i]
        if np.all(low[i - left : i] > val) and np.all(low[i + 1 : i + right + 1] >= val):
            idxs.append(i)
    return idxs


def run_bat_engine(
    df: pd.DataFrame,
    symbol: str,
    coin: str,
    timeframe: str,
    params: Dict,
) -> BatScanResult:
    if df is None or len(df) < max(params["trend_len"], params["pivot_left"] + params["pivot_right"] + 5):
        return BatScanResult(
            symbol=symbol,
            coin=coin,
            timeframe=timeframe,
            state=0,
            status_label="IDLE",
            sig_code="-",
            pattern_type="",
            overstretched=False,
            conf_score=0,
            conf_pct=0.0,
            last_close=float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else None,
        )

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    atr = compute_atr(df, params["atr_len"]).to_numpy(dtype=float)
    sigma_basis = df["close"].rolling(params["sigma_len"], min_periods=1).mean().to_numpy(dtype=float)
    sigma_dev = (
        df["close"].rolling(params["sigma_len"], min_periods=1).std(ddof=0).fillna(0.0).to_numpy(dtype=float)
        * params["sigma_mult"]
    )
    upper_sigma = sigma_basis + sigma_dev
    lower_sigma = sigma_basis - sigma_dev

    rsi_series = compute_rsi(df["close"], params["rsi_len"])
    stoch_k_s, stoch_d_s = compute_stoch_rsi(
        rsi_series, params["stoch_len"], params["smooth_k"], params["smooth_d"]
    )
    ema_trend = df["close"].ewm(span=params["trend_len"], adjust=False).mean().to_numpy(dtype=float)

    rsi = rsi_series.to_numpy(dtype=float)
    stoch_k = stoch_k_s.to_numpy(dtype=float)
    stoch_d = stoch_d_s.to_numpy(dtype=float)

    pv_l = params["pivot_left"]
    pv_r = params["pivot_right"]
    atr_m = params["atr_mult"]

    piv_h_set = set(_pivot_highs(high, pv_l, pv_r))
    piv_l_set = set(_pivot_lows(low, pv_l, pv_r))

    high_idx_arr: List[int] = []
    high_prc_arr: List[float] = []
    high_rsi_arr: List[float] = []

    low_idx_arr: List[int] = []
    low_prc_arr: List[float] = []
    low_rsi_arr: List[float] = []

    internal_state = 0
    pattern_type = ""
    x_idx = a_idx = b_idx = c_idx = -1
    x_price = a_price = b_price = c_price = d_price = np.nan
    price_sl = price_tp1 = price_tp2 = price_tp3 = b_rsi = np.nan
    tp1_hit = False
    tp2_hit = False

    for bar_idx in range(len(df)):
        cand_idx = bar_idx - pv_r
        if cand_idx >= 0:
            if cand_idx in piv_h_set:
                high_idx_arr.insert(0, cand_idx)
                high_prc_arr.insert(0, float(high[cand_idx]))
                high_rsi_arr.insert(0, float(rsi[cand_idx]))
                if len(high_idx_arr) > 6:
                    high_idx_arr.pop()
                    high_prc_arr.pop()
                    high_rsi_arr.pop()
            if cand_idx in piv_l_set:
                low_idx_arr.insert(0, cand_idx)
                low_prc_arr.insert(0, float(low[cand_idx]))
                low_rsi_arr.insert(0, float(rsi[cand_idx]))
                if len(low_idx_arr) > 6:
                    low_idx_arr.pop()
                    low_prc_arr.pop()
                    low_rsi_arr.pop()

        cur_atr = float(atr[bar_idx])
        cur_high = float(high[bar_idx])
        cur_low = float(low[bar_idx])

        if internal_state in (0, 3):
            if len(low_prc_arr) >= 2 and len(high_prc_arr) >= 2:
                px_i, px_p = low_idx_arr[1], low_prc_arr[1]
                pa_i, pa_p = high_idx_arr[1], high_prc_arr[1]
                pb_i, pb_p = low_idx_arr[0], low_prc_arr[0]
                pc_i, pc_p = high_idx_arr[0], high_prc_arr[0]
                if px_i < pa_i < pb_i < pc_i:
                    xa_h = pa_p - px_p
                    b_ret = (pa_p - pb_p) / xa_h if xa_h != 0 else 0.0
                    ab_h = pa_p - pb_p
                    c_ret = (pc_p - pb_p) / ab_h if ab_h != 0 else 0.0
                    if 0.382 <= b_ret <= 0.618 and c_ret >= 0.382 and pc_p < pa_p:
                        internal_state = 1
                        pattern_type = "BULLISH"
                        x_idx, x_price = px_i, px_p
                        a_idx, a_price = pa_i, pa_p
                        b_idx, b_price = pb_i, pb_p
                        c_idx, c_price = pc_i, pc_p
                        b_rsi = low_rsi_arr[0]
                        tp1_hit = False
                        tp2_hit = False
                        d_price = a_price - (0.886 * xa_h)
                        price_sl = (a_price - (1.113 * xa_h)) - (cur_atr * atr_m)
                        price_tp1 = price_tp2 = price_tp3 = np.nan

            if internal_state == 0 and len(high_prc_arr) >= 2 and len(low_prc_arr) >= 2:
                px_i, px_p = high_idx_arr[1], high_prc_arr[1]
                pa_i, pa_p = low_idx_arr[1], low_prc_arr[1]
                pb_i, pb_p = high_idx_arr[0], high_prc_arr[0]
                pc_i, pc_p = low_idx_arr[0], low_prc_arr[0]
                if px_i < pa_i < pb_i < pc_i:
                    xa_h = px_p - pa_p
                    b_ret = (pb_p - pa_p) / xa_h if xa_h != 0 else 0.0
                    ab_h = pb_p - pa_p
                    c_ret = (pb_p - pc_p) / ab_h if ab_h != 0 else 0.0
                    if 0.382 <= b_ret <= 0.618 and c_ret >= 0.382 and pc_p > pa_p:
                        internal_state = 1
                        pattern_type = "BEARISH"
                        x_idx, x_price = px_i, px_p
                        a_idx, a_price = pa_i, pa_p
                        b_idx, b_price = pb_i, pb_p
                        c_idx, c_price = pc_i, pc_p
                        b_rsi = high_rsi_arr[0]
                        tp1_hit = False
                        tp2_hit = False
                        d_price = a_price + (0.886 * xa_h)
                        price_sl = (a_price + (1.113 * xa_h)) + (cur_atr * atr_m)
                        price_tp1 = price_tp2 = price_tp3 = np.nan

        if internal_state == 1:
            if pattern_type == "BULLISH":
                if cur_high > a_price:
                    internal_state = 3
                elif cur_low <= price_sl:
                    internal_state = 3
                elif cur_low <= d_price:
                    internal_state = 2
                    ad_h = abs(a_price - d_price)
                    price_tp1 = d_price + (0.382 * ad_h)
                    price_tp2 = d_price + (0.500 * ad_h)
                    price_tp3 = d_price + (0.618 * ad_h)
            elif pattern_type == "BEARISH":
                if cur_low < a_price:
                    internal_state = 3
                elif cur_high >= price_sl:
                    internal_state = 3
                elif cur_high >= d_price:
                    internal_state = 2
                    ad_h = abs(d_price - a_price)
                    price_tp1 = d_price - (0.382 * ad_h)
                    price_tp2 = d_price - (0.500 * ad_h)
                    price_tp3 = d_price - (0.618 * ad_h)

        if internal_state == 2:
            if pattern_type == "BULLISH":
                if cur_high >= price_tp1:
                    tp1_hit = True
                if cur_high >= price_tp2:
                    tp2_hit = True
                if cur_high >= price_tp3 or cur_low <= price_sl:
                    internal_state = 0
            elif pattern_type == "BEARISH":
                if cur_low <= price_tp1:
                    tp1_hit = True
                if cur_low <= price_tp2:
                    tp2_hit = True
                if cur_low <= price_tp3 or cur_high >= price_sl:
                    internal_state = 0

    last_i = len(df) - 1
    cur_close = float(close[last_i])
    cur_rsi = float(rsi[last_i])
    cur_k = float(stoch_k[last_i])
    cur_d = float(stoch_d[last_i])
    cur_ema = float(ema_trend[last_i])

    overstretched = False
    if internal_state == 1:
        if pattern_type == "BULLISH" and cur_close < float(lower_sigma[last_i]):
            overstretched = True
        elif pattern_type == "BEARISH" and cur_close > float(upper_sigma[last_i]):
            overstretched = True

    conf_score = 0
    bad_confluences: List[str] = []
    if internal_state in (1, 2):
        rsi_div = False
        stoch_ok = False
        trend_ok = False
        if pattern_type == "BULLISH":
            rsi_div = (not np.isnan(b_rsi)) and (d_price <= b_price) and (cur_rsi > b_rsi)
            stoch_ok = (cur_k < 30.0) and (cur_k > cur_d)
            trend_ok = cur_close > cur_ema
            if not rsi_div:
                bad_confluences.append("RSI Div Weak/Broken")
            if not stoch_ok:
                bad_confluences.append("Stoch RSI Not Bullish")
            if not trend_ok:
                bad_confluences.append("Below Trend EMA")
        elif pattern_type == "BEARISH":
            rsi_div = (not np.isnan(b_rsi)) and (d_price >= b_price) and (cur_rsi < b_rsi)
            stoch_ok = (cur_k > 70.0) and (cur_k < cur_d)
            trend_ok = cur_close < cur_ema
            if not rsi_div:
                bad_confluences.append("RSI Div Weak/Broken")
            if not stoch_ok:
                bad_confluences.append("Stoch RSI Not Bearish")
            if not trend_ok:
                bad_confluences.append("Above Trend EMA")
        conf_score = int(rsi_div) + int(stoch_ok) + int(trend_ok)

    conf_pct = round((conf_score / 3.0) * 100.0, 1)

    if internal_state == 1:
        status_label = "READY TO BE"
        base = "W" if overstretched else "R"
        sig_code = f"{base}{conf_score}"
    elif internal_state == 2:
        status_label = "CONFIRMED"
        sig_code = f"C{conf_score}"
    else:
        status_label = "IDLE"
        sig_code = "-"

    zone_low: Optional[float] = None
    zone_high: Optional[float] = None
    zone_label = "-"
    beyond_optimal_entry = False
    optimal_entry_status = "-"

    if internal_state in (1, 2) and not np.isnan(d_price) and not np.isnan(price_sl):
        zone_low = float(min(d_price, price_sl))
        zone_high = float(max(d_price, price_sl))
        zone_label = "Demand Zone (Reversal)" if pattern_type == "BULLISH" else "Supply Zone (Reversal)"

        tol = max(abs(float(d_price)) * 0.004, float(atr[last_i]) * 0.2)
        if pattern_type == "BULLISH":
            if cur_close < float(d_price) - tol:
                beyond_optimal_entry = True
                optimal_entry_status = "Beyond Optimal Entry (Below D)"
            elif abs(cur_close - float(d_price)) <= tol:
                optimal_entry_status = "In Optimal Entry Zone"
            else:
                optimal_entry_status = "Approaching D" if internal_state == 1 else "Above D (Rebounding)"
        elif pattern_type == "BEARISH":
            if cur_close > float(d_price) + tol:
                beyond_optimal_entry = True
                optimal_entry_status = "Beyond Optimal Entry (Above D)"
            elif abs(cur_close - float(d_price)) <= tol:
                optimal_entry_status = "In Optimal Entry Zone"
            else:
                optimal_entry_status = "Approaching D" if internal_state == 1 else "Below D (Pulling Back)"

    pivots: Dict[str, Tuple[int, float]] = {}
    leg_statuses: Dict[str, str] = {}
    if internal_state in (1, 2) and x_idx >= 0:
        pivots = {
            "X": (x_idx, float(x_price)),
            "A": (a_idx, float(a_price)),
            "B": (b_idx, float(b_price)),
            "C": (c_idx, float(c_price)),
        }
        leg_statuses = {
            "Leg X -> A": "FORMED",
            "Leg A -> B": "FORMED",
            "Leg B -> C": "FORMED",
            "Leg C -> D": "FORMED" if internal_state == 2 else "EXPECTED",
        }

    return BatScanResult(
        symbol=symbol,
        coin=coin,
        timeframe=timeframe,
        state=internal_state,
        status_label=status_label,
        sig_code=sig_code,
        pattern_type=pattern_type,
        overstretched=overstretched,
        conf_score=conf_score,
        conf_pct=conf_pct,
        bad_confluences=bad_confluences,
        entry_d=None if np.isnan(d_price) else float(d_price),
        sl=None if np.isnan(price_sl) else float(price_sl),
        tp1=None if np.isnan(price_tp1) else float(price_tp1),
        tp2=None if np.isnan(price_tp2) else float(price_tp2),
        tp3=None if np.isnan(price_tp3) else float(price_tp3),
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        zone_low=zone_low,
        zone_high=zone_high,
        zone_label=zone_label,
        beyond_optimal_entry=beyond_optimal_entry,
        optimal_entry_status=optimal_entry_status,
        last_close=cur_close,
        pivots=pivots,
        leg_statuses=leg_statuses,
    )
