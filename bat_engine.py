from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from market_data import TF_ORDER, compute_atr, compute_rsi, compute_stoch_rsi


@dataclass
class BatScanResult:
    symbol: str
    coin: str
    timeframe: str
    state: int
    status_label: str
    status_desc: str
    error_msg: str
    sig_code: str
    pattern_type: str
    overstretched: bool
    conf_score: int
    conf_pct: float
    total_score: int
    score_adjustments: List[str] = field(default_factory=list)
    bad_confluences: List[str] = field(default_factory=list)
    mtf_notes: List[str] = field(default_factory=list)
    entry_d: Optional[float] = None
    optimal_zone_low: Optional[float] = None
    optimal_zone_high: Optional[float] = None
    sl: Optional[float] = None
    secondary_sl: Optional[float] = None
    break_even_sl: Optional[float] = None
    bias_change_level: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    tp_obstacle_note: str = "Clear path"
    zone_low: Optional[float] = None
    zone_high: Optional[float] = None
    zone_label: str = "-"
    beyond_optimal_entry: bool = False
    optimal_entry_status: str = "-"
    last_close: Optional[float] = None
    pivots: Dict[str, Tuple[int, float]] = field(default_factory=dict)
    leg_statuses: Dict[str, str] = field(default_factory=dict)


def _pine_pivothigh(high: np.ndarray, left: int, right: int) -> List[int]:
    idxs: List[int] = []
    n = len(high)
    for i in range(left, n - right):
        val = high[i]
        if np.all(high[i - left : i] <= val) and np.all(high[i + 1 : i + right + 1] <= val):
            if np.any(high[i - left : i] < val) or np.any(high[i + 1 : i + right + 1] < val):
                idxs.append(i)
    return idxs


def _pine_pivotlow(low: np.ndarray, left: int, right: int) -> List[int]:
    idxs: List[int] = []
    n = len(low)
    for i in range(left, n - right):
        val = low[i]
        if np.all(low[i - left : i] >= val) and np.all(low[i + 1 : i + right + 1] >= val):
            if np.any(low[i - left : i] > val) or np.any(low[i + 1 : i + right + 1] > val):
                idxs.append(i)
    return idxs


def compute_dynamic_secondary_sl(
    direction: str,
    live_price: Optional[float],
    entry_d: Optional[float],
    primary_sl: Optional[float],
    tp1: Optional[float],
    atr_val: float,
    recent_swing_low: Optional[float],
    recent_swing_high: Optional[float],
) -> Optional[float]:
    if primary_sl is None or entry_d is None or live_price is None:
        return primary_sl

    if direction == "BULLISH":
        candidate = primary_sl
        if tp1 is not None and live_price >= tp1:
            candidate = max(candidate, entry_d + (0.15 * max(0.0, (tp1 - entry_d))))
        elif live_price > entry_d:
            trail = live_price - (1.25 * atr_val)
            if recent_swing_low is not None and recent_swing_low < live_price:
                trail = max(trail, recent_swing_low - (0.25 * atr_val))
            candidate = max(candidate, trail)
        return float(candidate)

    if direction == "BEARISH":
        candidate = primary_sl
        if tp1 is not None and live_price <= tp1:
            candidate = min(candidate, entry_d - (0.15 * max(0.0, (entry_d - tp1))))
        elif live_price < entry_d:
            trail = live_price + (1.25 * atr_val)
            if recent_swing_high is not None and recent_swing_high > live_price:
                trail = min(trail, recent_swing_high + (0.25 * atr_val))
            candidate = min(candidate, trail)
        return float(candidate)

    return primary_sl


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
            status_desc="Hunting Waves",
            error_msg="",
            sig_code="-",
            pattern_type="",
            overstretched=False,
            conf_score=0,
            conf_pct=0.0,
            total_score=0,
            last_close=float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else None,
        )

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.ones(len(df))

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

    piv_h_set = set(_pine_pivothigh(high, pv_l, pv_r))
    piv_l_set = set(_pine_pivotlow(low, pv_l, pv_r))

    high_idx_arr: List[int] = []
    high_prc_arr: List[float] = []
    high_rsi_arr: List[float] = []

    low_idx_arr: List[int] = []
    low_prc_arr: List[float] = []
    low_rsi_arr: List[float] = []

    internal_state = 0
    pattern_type = ""
    error_msg = ""
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

        if internal_state == 0 or internal_state == 3:
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
                        error_msg = ""
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
                        error_msg = ""
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
                    error_msg = "C Crossed Point A High"
                elif cur_low <= price_sl:
                    internal_state = 3
                    error_msg = "Price Violated Stop Loss"
                elif cur_low <= d_price:
                    internal_state = 2
                    ad_h = abs(a_price - d_price)
                    price_tp1 = d_price + (0.382 * ad_h)
                    price_tp2 = d_price + (0.500 * ad_h)
                    price_tp3 = d_price + (0.618 * ad_h)
            elif pattern_type == "BEARISH":
                if cur_low < a_price:
                    internal_state = 3
                    error_msg = "C Crossed Point A Low"
                elif cur_high >= price_sl:
                    internal_state = 3
                    error_msg = "Price Violated Stop Loss"
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
    cur_atr = float(atr[last_i])
    avg_atr = float(np.nanmean(atr[max(0, last_i - 30) : last_i + 1])) or cur_atr

    vol_recent = float(np.nanmean(volume[max(0, last_i - 5) : last_i + 1]))
    vol_base = float(np.nanmean(volume[max(0, last_i - 30) : last_i + 1])) or 1.0
    vol_ratio = vol_recent / vol_base

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
        status_desc = "WAIT (Overstretched)" if overstretched else "READY (Execute)"
        base = "W" if overstretched else "R"
        sig_code = f"{base}{conf_score}"
    elif internal_state == 2:
        status_label = "CONFIRMED"
        status_desc = f"{pattern_type} ACTIVE"
        sig_code = f"C{conf_score}"
    elif internal_state == 3:
        status_label = "IDLE"
        status_desc = f"INVALID: {error_msg}" if error_msg else "INVALID"
        sig_code = "-"
    else:
        status_label = "IDLE"
        status_desc = "Hunting Waves"
        sig_code = "-"

    piv_h_list = sorted(piv_h_set)
    piv_l_list = sorted(piv_l_set)
    supports = [float(low[i]) for i in piv_l_list if float(low[i]) < cur_close]
    resistances = [float(high[i]) for i in piv_h_list if float(high[i]) > cur_close]
    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    zone_low: Optional[float] = None
    zone_high: Optional[float] = None
    optimal_zone_low: Optional[float] = None
    optimal_zone_high: Optional[float] = None
    secondary_sl: Optional[float] = None
    break_even_sl: Optional[float] = None
    bias_change_level: Optional[float] = None
    tp_obstacle_note = "Clear path"
    zone_label = "-"
    beyond_optimal_entry = False
    optimal_entry_status = "-"

    score_adjustments: List[str] = []
    total_score = 45 + (conf_score * 15) if internal_state in (1, 2) else 0

    if internal_state in (1, 2) and not np.isnan(d_price) and not np.isnan(price_sl):
        d_f = float(d_price)
        sl_f = float(price_sl)
        tp1_f = None if np.isnan(price_tp1) else float(price_tp1)
        zone_low = float(min(d_f, sl_f))
        zone_high = float(max(d_f, sl_f))
        zone_label = "Demand Zone (Reversal)" if pattern_type == "BULLISH" else "Supply Zone (Reversal)"

        opt_pad = max(abs(d_f) * 0.0035, cur_atr * 0.25)
        optimal_zone_low = d_f - opt_pad
        optimal_zone_high = d_f + opt_pad

        secondary_sl = compute_dynamic_secondary_sl(
            direction=pattern_type,
            live_price=cur_close,
            entry_d=d_f,
            primary_sl=sl_f,
            tp1=tp1_f,
            atr_val=cur_atr,
            recent_swing_low=nearest_support,
            recent_swing_high=nearest_resistance,
        )

        if pattern_type == "BULLISH":
            be_buf = max(abs(d_f) * 0.0005, cur_atr * 0.05)
            break_even_sl = float(d_f + be_buf)
            bias_change_level = float(min(b_price, cur_ema)) if not np.isnan(b_price) else cur_ema
            if cur_close < d_f - opt_pad:
                beyond_optimal_entry = True
                optimal_entry_status = "Beyond Optimal Entry (Below D)"
            elif abs(cur_close - d_f) <= opt_pad:
                optimal_entry_status = "In Optimal Reversal Zone"
            else:
                optimal_entry_status = "Wait for Optimal Reversal Zone (Above D)"

            if nearest_resistance is not None and tp1_f is not None and nearest_resistance < tp1_f:
                tp_obstacle_note = f"Resistance wall @ {nearest_resistance:.6f} before TP1"
                total_score -= 8
                score_adjustments.append("-8 (Resistance before TP1)")

        elif pattern_type == "BEARISH":
            be_buf = max(abs(d_f) * 0.0005, cur_atr * 0.05)
            break_even_sl = float(d_f - be_buf)
            bias_change_level = float(max(b_price, cur_ema)) if not np.isnan(b_price) else cur_ema
            if cur_close > d_f + opt_pad:
                beyond_optimal_entry = True
                optimal_entry_status = "Beyond Optimal Entry (Above D)"
            elif abs(cur_close - d_f) <= opt_pad:
                optimal_entry_status = "In Optimal Reversal Zone"
            else:
                optimal_entry_status = "Wait for Optimal Reversal Zone (Below D)"

            if nearest_support is not None and tp1_f is not None and nearest_support > tp1_f:
                tp_obstacle_note = f"Support wall @ {nearest_support:.6f} before TP1"
                total_score -= 8
                score_adjustments.append("-8 (Support before TP1)")

        if cur_atr > avg_atr * 1.35:
            total_score -= 6
            score_adjustments.append("-6 (High ATR Volatility)")
        elif cur_atr <= avg_atr * 1.1:
            total_score += 5
            score_adjustments.append("+5 (Controlled ATR Volatility)")

        if vol_ratio >= 1.25:
            total_score += 6
            score_adjustments.append("+6 (Strong Volume)")
        elif vol_ratio < 0.7:
            total_score -= 5
            score_adjustments.append("-5 (Low Volume)")

    total_score = int(max(0, min(100, total_score)))

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
        status_desc=status_desc,
        error_msg=error_msg,
        sig_code=sig_code,
        pattern_type=pattern_type,
        overstretched=overstretched,
        conf_score=conf_score,
        conf_pct=conf_pct,
        total_score=total_score,
        score_adjustments=score_adjustments,
        bad_confluences=bad_confluences,
        entry_d=None if np.isnan(d_price) else float(d_price),
        optimal_zone_low=optimal_zone_low,
        optimal_zone_high=optimal_zone_high,
        sl=None if np.isnan(price_sl) else float(price_sl),
        secondary_sl=secondary_sl,
        break_even_sl=break_even_sl,
        bias_change_level=bias_change_level,
        tp1=None if np.isnan(price_tp1) else float(price_tp1),
        tp2=None if np.isnan(price_tp2) else float(price_tp2),
        tp3=None if np.isnan(price_tp3) else float(price_tp3),
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        tp_obstacle_note=tp_obstacle_note,
        zone_low=zone_low,
        zone_high=zone_high,
        zone_label=zone_label,
        beyond_optimal_entry=beyond_optimal_entry,
        optimal_entry_status=optimal_entry_status,
        last_close=cur_close,
        pivots=pivots,
        leg_statuses=leg_statuses,
    )


def enrich_multi_timeframe_context(results: List[BatScanResult]) -> List[BatScanResult]:
    by_coin: Dict[str, Dict[str, BatScanResult]] = {}
    for r in results:
        by_coin.setdefault(r.coin, {})[r.timeframe] = r

    for r in results:
        if r.status_label == "IDLE":
            continue
        tf_map = by_coin.get(r.coin, {})
        cur_rank = TF_ORDER.index(r.timeframe) if r.timeframe in TF_ORDER else 0
        higher_tfs = [t for t in TF_ORDER[cur_rank + 1 :] if t in tf_map]

        notes: List[str] = []
        adj = r.total_score

        for htf in higher_tfs:
            hr = tf_map[htf]
            if hr.status_label == "READY TO BE" and hr.pattern_type == r.pattern_type:
                notes.append(f"{htf} {hr.pattern_type} READY TO BE (D: {hr.entry_d})")
                if r.status_label == "CONFIRMED":
                    adj += 5
                    r.score_adjustments.append(f"+5 ({htf} Same-Dir Forming)")
            elif hr.status_label == "READY TO BE" and hr.pattern_type != r.pattern_type and hr.pattern_type:
                notes.append(f"{htf} {hr.pattern_type} READY TO BE (D: {hr.entry_d})")
                adj -= 7
                r.score_adjustments.append(f"-7 ({htf} Opposing Forming)")
            elif hr.status_label == "CONFIRMED" and hr.pattern_type == r.pattern_type:
                notes.append(f"{htf} {hr.pattern_type} CONFIRMED")
                adj += 6
                r.score_adjustments.append(f"+6 ({htf} Confirmed Alignment)")

        if not notes:
            notes.append("No conflicting higher-timeframe forming pattern")

        r.mtf_notes = notes
        r.total_score = int(max(0, min(100, adj)))

    return results
