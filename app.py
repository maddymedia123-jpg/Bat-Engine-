from __future__ import annotations

import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from bat_engine import BatScanResult, enrich_multi_timeframe_context, run_bat_engine
from logbook import HORIZONS, filter_logbook_records, load_logbook, record_generated_signals
from market_data import (
    DEFAULT_20_SYMBOLS,
    TF_HOURS,
    TIMEFRAME_OPTIONS,
    TV_INTERVAL_MAP,
    clean_coin_name,
    compute_price_correlation,
    fetch_kucoin_klines,
    fetch_kucoin_last_price,
    fetch_liquidation_snapshot,
    fetch_web_reference_prices,
    format_eta_pkt,
    generate_layman_ai_summary,
    to_kucoin_symbol,
)

st.set_page_config(page_title="Harmonics Engine — Bat Pattern", layout="wide")


def fmt_price(val: Optional[float], scale: float = 1.0) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    adj = val * scale
    return f"${adj:.6f}".rstrip("0").rstrip(".")


def compute_tp_progress(
    live_price: Optional[float],
    entry: Optional[float],
    tp: Optional[float],
    sl: Optional[float],
    direction: str,
) -> float:
    if live_price is None or entry is None or tp is None or tp == entry:
        return 0.0
    if direction == "BULLISH":
        if sl is not None and live_price <= sl:
            return 0.0
        pct = ((live_price - entry) / (tp - entry)) * 100.0
    else:
        if sl is not None and live_price >= sl:
            return 0.0
        pct = ((entry - live_price) / (entry - tp)) * 100.0
    return float(max(0.0, min(100.0, pct)))


def render_tradingview_chart(tv_symbol: str, tf_label: str, height: int = 430) -> None:
    interval = TV_INTERVAL_MAP.get(tf_label, "60")
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_harmonics_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%",
          "height": {height},
          "symbol": "{tv_symbol}",
          "interval": "{interval}",
          "timezone": "Asia/Karachi",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#1a2027",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "container_id": "tv_harmonics_chart"
        }});
      </script>
    </div>
    """
    components.html(html, height=height + 16)


def render_pattern_mapping_chart(
    df: pd.DataFrame,
    res: BatScanResult,
    liq_info: Dict[str, Optional[float] | str],
    scale: float = 1.0,
) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    plot_df = df.tail(120).copy()
    offset = len(df) - len(plot_df)
    x_vals = list(range(len(plot_df)))

    fig.add_trace(
        go.Candlestick(
            x=x_vals,
            open=plot_df["open"] * scale,
            high=plot_df["high"] * scale,
            low=plot_df["low"] * scale,
            close=plot_df["close"] * scale,
            name=f"{res.coin} {res.timeframe}",
        )
    )

    if res.pivots and all(k in res.pivots for k in ("X", "A", "B", "C")):
        px_i, px_v = res.pivots["X"][0] - offset, res.pivots["X"][1] * scale
        pa_i, pa_v = res.pivots["A"][0] - offset, res.pivots["A"][1] * scale
        pb_i, pb_v = res.pivots["B"][0] - offset, res.pivots["B"][1] * scale
        pc_i, pc_v = res.pivots["C"][0] - offset, res.pivots["C"][1] * scale

        formed_x = [i for i in (px_i, pa_i, pb_i, pc_i) if i >= 0]
        formed_y = [v for i, v in zip((px_i, pa_i, pb_i, pc_i), (px_v, pa_v, pb_v, pc_v)) if i >= 0]
        line_col = "#00C853" if res.pattern_type == "BULLISH" else "#FF3B3B"
        if len(formed_x) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=formed_x,
                    y=formed_y,
                    mode="lines+markers+text",
                    name="X-A-B-C Legs",
                    text=["X", "A", "B", "C"][: len(formed_x)],
                    textposition="top center",
                    line=dict(color=line_col, width=2.5),
                )
            )

        if res.entry_d is not None and pc_i >= 0:
            end_x = len(plot_df) - 1
            fig.add_trace(
                go.Scatter(
                    x=[pc_i, end_x],
                    y=[pc_v, res.entry_d * scale],
                    mode="lines",
                    name="C -> D Leg",
                    line=dict(
                        color="#FFD54F" if res.status_label == "READY TO BE" else line_col,
                        width=2,
                        dash="dash" if res.status_label == "READY TO BE" else "solid",
                    ),
                )
            )

    if res.zone_low is not None and res.zone_high is not None:
        fig.add_hrect(
            y0=res.zone_low * scale,
            y1=res.zone_high * scale,
            fillcolor="rgba(0, 200, 83, 0.15)" if res.pattern_type == "BULLISH" else "rgba(255, 59, 59, 0.15)",
            line_width=1,
            line_dash="dot",
            annotation_text=res.zone_label,
            annotation_position="top left",
        )

    if res.entry_d is not None:
        fig.add_hline(y=res.entry_d * scale, line_dash="dash", line_color="#FFD54F", annotation_text="Entry (D)")
    if res.sl is not None:
        fig.add_hline(y=res.sl * scale, line_dash="solid", line_color="#FF3B3B", annotation_text="Primary SL")
    if res.suggested_sl is not None:
        fig.add_hline(y=res.suggested_sl * scale, line_dash="dashdot", line_color="#FF9800", annotation_text="Wick-Buffer SL")
    if res.bias_change_level is not None:
        fig.add_hline(y=res.bias_change_level * scale, line_dash="longdash", line_color="#AB47BC", annotation_text="Bias-Change")
    if res.nearest_support is not None:
        fig.add_hline(y=res.nearest_support * scale, line_dash="dot", line_color="#29B6F6", annotation_text="Nearest Support")
    if res.nearest_resistance is not None:
        fig.add_hline(y=res.nearest_resistance * scale, line_dash="dot", line_color="#FFA726", annotation_text="Nearest Resistance")

    upper_liq = liq_info.get("upper_liq")
    lower_liq = liq_info.get("lower_liq")
    if isinstance(upper_liq, (int, float)):
        fig.add_hline(y=float(upper_liq) * scale, line_dash="dot", line_color="#EF5350", annotation_text="Est. Liq High")
    if isinstance(lower_liq, (int, float)):
        fig.add_hline(y=float(lower_liq) * scale, line_dash="dot", line_color="#26A69A", annotation_text="Est. Liq Low")

    for tp_val, label in ((res.tp1, "TP1"), (res.tp2, "TP2"), (res.tp3, "TP3")):
        if tp_val is not None:
            fig.add_hline(y=tp_val * scale, line_dash="dot", line_color="#3DDC84", annotation_text=label)

    fig.update_layout(
        template="plotly_dark",
        height=430,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(DEFAULT_20_SYMBOLS)
if "active_trades" not in st.session_state:
    st.session_state.active_trades = []
if "logbook" not in st.session_state:
    st.session_state.logbook = load_logbook()
if "ai_summaries" not in st.session_state:
    st.session_state.ai_summaries = {}

with st.sidebar:
    st.header("Watchlist & Timeframes (20+ Coins)")

    add_col, del_col = st.columns(2)
    with add_col:
        new_coin_raw = st.text_input("Add Coin", placeholder="e.g. LINK or LINK-USDT")
        if st.button("Add Coin", use_container_width=True) and new_coin_raw.strip():
            formatted = to_kucoin_symbol(new_coin_raw.strip())
            if formatted not in st.session_state.watchlist:
                st.session_state.watchlist.append(formatted)
                st.rerun()

    with del_col:
        del_target = st.selectbox("Delete Coin", options=["-"] + st.session_state.watchlist)
        if st.button("Delete Coin", use_container_width=True) and del_target != "-":
            st.session_state.watchlist = [c for c in st.session_state.watchlist if c != del_target]
            st.rerun()

    st.caption(f"Active Watchlist Count: {len(st.session_state.watchlist)} coins")
    selected_tfs = st.multiselect(
        "Timeframes (Initial: 1H, 2H, 4H, 1D)",
        options=list(TIMEFRAME_OPTIONS.keys()),
        default=["1H", "2H", "4H", "1D"],
    )

    st.divider()
    st.subheader("Background Live Scanning")
    auto_refresh = st.checkbox("Auto-refresh continuously", value=True)
    refresh_sec = st.slider("Refresh interval (sec)", min_value=15, max_value=180, value=45, step=15)

    st.divider()
    st.subheader("Display & AI Summary")
    price_mode = st.radio(
        "Display levels in:",
        options=["KuCoin Native", "Web / Reference Adjusted"],
        index=0,
    )
    show_overstretched = st.checkbox("Show overstretched READY TO BE (W) rows", value=False)
    ai_api_key = st.text_input("AI API Key (Optional)", type="password", placeholder="sk-...")

    with st.expander("Pine Bat Engine Parameters"):
        pivot_left = st.number_input("Pivot Left Strength", min_value=1, max_value=20, value=4)
        pivot_right = st.number_input("Pivot Right Strength", min_value=1, max_value=20, value=4)
        rsi_len = st.number_input("RSI Length", min_value=1, max_value=100, value=14)
        stoch_len = st.number_input("Stoch RSI Length", min_value=1, max_value=100, value=14)
        smooth_k = st.number_input("Stoch RSI Smooth K", min_value=1, max_value=20, value=3)
        smooth_d = st.number_input("Stoch RSI Smooth D", min_value=1, max_value=20, value=3)
        trend_len = st.number_input("Trend EMA Length", min_value=1, max_value=500, value=100)
        atr_len = st.number_input("ATR Length for SL", min_value=1, max_value=100, value=14)
        atr_mult = st.number_input("ATR SL Multiplier Buffer", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        sigma_len = st.number_input("Sigma Band Filter Length", min_value=1, max_value=100, value=20)
        sigma_mult = st.number_input("Sigma Overstretch Multiplier", min_value=0.5, max_value=10.0, value=2.5, step=0.1)

    if st.button("Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

symbols_list = st.session_state.watchlist if st.session_state.watchlist else list(DEFAULT_20_SYMBOLS)
if not selected_tfs:
    selected_tfs = ["1H", "2H", "4H", "1D"]

params = {
    "pivot_left": int(pivot_left),
    "pivot_right": int(pivot_right),
    "rsi_len": int(rsi_len),
    "stoch_len": int(stoch_len),
    "smooth_k": int(smooth_k),
    "smooth_d": int(smooth_d),
    "trend_len": int(trend_len),
    "atr_len": int(atr_len),
    "atr_mult": float(atr_mult),
    "sigma_len": int(sigma_len),
    "sigma_mult": float(sigma_mult),
}

st.title("Harmonics Engine")
st.caption("Multi-timeframe harmonic scanner with Bat Pattern tab, active trade tracking (PKT), stacked charts, and backtest logbook.")

coins_tuple = tuple(clean_coin_name(s) for s in symbols_list)
web_prices = fetch_web_reference_prices(coins_tuple)

corr_by_coin: Dict[str, Dict[str, Optional[float]]] = {}
for sym in symbols_list:
    coin = clean_coin_name(sym)
    kc_last = None
    try:
        kc_last = fetch_kucoin_last_price(sym)
    except Exception:
        kc_last = None
    corr_by_coin[coin] = compute_price_correlation(kc_last, web_prices.get(coin))

results: List[BatScanResult] = []
df_lookup: Dict[tuple, pd.DataFrame] = {}

for tf in selected_tfs:
    for sym in symbols_list:
        coin = clean_coin_name(sym)
        try:
            df = fetch_kucoin_klines(sym, tf, limit=250)
            if corr_by_coin[coin]["kucoin"] is None and not df.empty:
                corr_by_coin[coin] = compute_price_correlation(float(df["close"].iloc[-1]), web_prices.get(coin))
        except Exception as exc:
            st.warning(f"KuCoin fetch issue for {coin} ({tf}): {exc}")
            df = pd.DataFrame()
        df_lookup[(coin, tf)] = df
        results.append(run_bat_engine(df, sym, coin, tf, params))

results = enrich_multi_timeframe_context(results)
st.session_state.logbook = record_generated_signals(st.session_state.logbook, results)

result_map: Dict[tuple, BatScanResult] = {(r.coin, r.timeframe): r for r in results}

# Update active locked trades live
for trade in st.session_state.active_trades:
    coin = trade["coin"]
    tf = trade["timeframe"]
    direction = trade["pattern_type"]
    live_price = corr_by_coin.get(coin, {}).get("kucoin")
    if live_price is None:
        live_price = trade.get("last_price")
    trade["last_price"] = live_price

    latest_scan = result_map.get((coin, tf))
    if latest_scan:
        trade["conf_score"] = latest_scan.conf_score
        trade["conf_pct"] = latest_scan.conf_pct
        trade["total_score"] = latest_scan.total_score
        trade["bad_confluences"] = latest_scan.bad_confluences
        trade["beyond_optimal_entry"] = latest_scan.beyond_optimal_entry
        trade["optimal_entry_status"] = latest_scan.optimal_entry_status
        trade["suggested_sl"] = latest_scan.suggested_sl
        trade["bias_change_level"] = latest_scan.bias_change_level
        trade["mtf_notes"] = latest_scan.mtf_notes
        trade["tp_obstacle_note"] = latest_scan.tp_obstacle_note

    if live_price is not None and not trade.get("sl_hit", False):
        if direction == "BULLISH":
            if trade.get("tp1") is not None and live_price >= trade["tp1"]:
                trade["tp1_hit"] = True
            if trade.get("tp2") is not None and live_price >= trade["tp2"]:
                trade["tp2_hit"] = True
            if trade.get("tp3") is not None and live_price >= trade["tp3"]:
                trade["tp3_hit"] = True
            if trade.get("sl") is not None and live_price <= trade["sl"]:
                trade["sl_hit"] = True
        else:
            if trade.get("tp1") is not None and live_price <= trade["tp1"]:
                trade["tp1_hit"] = True
            if trade.get("tp2") is not None and live_price <= trade["tp2"]:
                trade["tp2_hit"] = True
            if trade.get("tp3") is not None and live_price <= trade["tp3"]:
                trade["tp3_hit"] = True
            if trade.get("sl") is not None and live_price >= trade["sl"]:
                trade["sl_hit"] = True

# 1. ACTIVE TRADE MONITORING TERMINAL (Always visible)
st.subheader(f"Active Trade Monitoring Terminal ({len(st.session_state.active_trades)} / 10 Locked)")
if st.session_state.active_trades:
    tf_tabs = st.tabs(["All Active Trades"] + [f"Active {t} Trades" for t in selected_tfs])
    for tab_idx, tab in enumerate(tf_tabs):
        with tab:
            subset = (
                st.session_state.active_trades
                if tab_idx == 0
                else [t for t in st.session_state.active_trades if t["timeframe"] == selected_tfs[tab_idx - 1]]
            )
            if not subset:
                st.info("No active trades in this filter.")
                continue

            for idx, trade in enumerate(subset):
                coin = trade["coin"]
                tf = trade["timeframe"]
                direction = trade["pattern_type"]
                live_p = trade.get("last_price")
                entry_p = trade.get("entry_d")
                sl_p = trade.get("sl")
                tp1_p, tp2_p, tp3_p = trade.get("tp1"), trade.get("tp2"), trade.get("tp3")
                sl_hit = bool(trade.get("sl_hit", False))
                any_tp_hit = bool(trade.get("tp1_hit") or trade.get("tp2_hit") or trade.get("tp3_hit"))
                tf_h = TF_HOURS.get(tf, 1.0)
                liq_snap = fetch_liquidation_snapshot(coin, live_p)

                with st.container(border=True):
                    c1, c2, c3, c4, c5 = st.columns([1.8, 1.8, 2.6, 2.0, 0.9])
                    with c1:
                        st.markdown(f"### {coin}/USDT · `{tf}`")
                        st.markdown(f"**{direction}** · `{trade.get('sig_code', 'C')}` ({trade.get('conf_pct', 0.0):.0f}%)")
                        st.caption(f"Score: **{trade.get('total_score', 0)}/100**")

                    with c2:
                        st.markdown(f"**Live:** {fmt_price(live_p)}")
                        st.markdown(f"**Entry:** {fmt_price(entry_p)}")
                        st.markdown(f"**Primary SL:** {fmt_price(sl_p)}")
                        st.caption(f"Wick-Buf SL: {fmt_price(trade.get('suggested_sl'))}")
                        st.caption(f"Bias-Change: {fmt_price(trade.get('bias_change_level'))}")

                    with c3:
                        if sl_hit and not any_tp_hit:
                            st.error("SL HIT — MONITORING FROZEN (TP bars cleared)")
                        else:
                            if sl_hit and any_tp_hit:
                                tp_done = [
                                    t
                                    for t, flag in (
                                        ("TP1", trade.get("tp1_hit")),
                                        ("TP2", trade.get("tp2_hit")),
                                        ("TP3", trade.get("tp3_hit")),
                                    )
                                    if flag
                                ]
                                st.warning(f"SL HIT AFTER {', '.join(tp_done)} — Keeping TP & SL status until killed.")

                            for tp_label, tp_val, hit_flag, mult_h in (
                                ("TP1", tp1_p, trade.get("tp1_hit", False), 4.0 * tf_h),
                                ("TP2", tp2_p, trade.get("tp2_hit", False), 8.0 * tf_h),
                                ("TP3", tp3_p, trade.get("tp3_hit", False), 16.0 * tf_h),
                            ):
                                prog = 100.0 if hit_flag else compute_tp_progress(live_p, entry_p, tp_val, sl_p, direction)
                                eta_str = "HIT" if hit_flag else format_eta_pkt(mult_h * (1.0 - prog / 100.0))
                                st.progress(int(round(prog)), text=f"{tp_label} ({fmt_price(tp_val)}): {prog:.0f}% · {eta_str}")

                    with c4:
                        if trade.get("beyond_optimal_entry"):
                            st.warning(f"Beyond Optimal Entry: {trade.get('optimal_entry_status')}")
                        else:
                            st.caption(f"Entry Zone Status: {trade.get('optimal_entry_status')}")

                        if trade.get("bad_confluences"):
                            st.error("Confluence Warning: " + ", ".join(trade["bad_confluences"]))
                        else:
                            st.success("Confluences Intact")

                        st.caption(f"TP Wall: {trade.get('tp_obstacle_note', 'Clear path')}")
                        st.caption(f"Liq: {liq_snap.get('liq_note', '-')}")

                    with c5:
                        if st.button("Kill", key=f"kill_{tab_idx}_{idx}_{coin}_{tf}", use_container_width=True):
                            st.session_state.active_trades = [
                                t
                                for t in st.session_state.active_trades
                                if not (t["coin"] == coin and t["timeframe"] == tf and t["pattern_type"] == direction)
                            ]
                            st.rerun()
else:
    st.info("No active trades locked yet. Lock a CONFIRMED signal from the Bat Pattern tab below.")

# 2. PATTERN TABS (Bat Pattern first; ready for extra patterns in separate tabs)
st.divider()
pattern_tabs = st.tabs(["Bat Pattern", "More Patterns (Soon)"])

with pattern_tabs[0]:
    col_ready, col_conf = st.columns(2)

    ready_items = [
        r for r in results if r.status_label == "READY TO BE" and (show_overstretched or not r.overstretched)
    ]
    conf_items = [r for r in results if r.status_label == "CONFIRMED"]

    def render_signal_panel(title: str, items: List[BatScanResult], is_confirmed: bool) -> None:
        st.subheader(title)
        if not items:
            st.info(f"No {title.lower()} setups right now.")
            return

        for idx, r in enumerate(items):
            scale = 1.0
            if price_mode == "Web / Reference Adjusted" and corr_by_coin[r.coin]["ratio"]:
                scale = float(corr_by_coin[r.coin]["ratio"])

            badge_color = "#00C853" if r.pattern_type == "BULLISH" else "#FF3B3B"
            with st.container(border=True):
                st.markdown(
                    f"<span style='color:{badge_color};font-weight:700;'>{r.coin}/USDT · {r.pattern_type} · [{r.timeframe}]</span> "
                    f"— **`{r.sig_code}`** | Conf: **{r.conf_pct:.0f}%** | Score: **{r.total_score}/100**",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Entry (D):** `{fmt_price(r.entry_d, scale)}` · "
                    f"**SL:** `{fmt_price(r.sl, scale)}` · "
                    f"**TP1:** `{fmt_price(r.tp1, scale)}` · "
                    f"**TP2/3:** `{fmt_price(r.tp2, scale)} / {fmt_price(r.tp3, scale)}`"
                )
                if r.zone_low is not None and r.zone_high is not None:
                    st.caption(
                        f"{r.zone_label}: {fmt_price(r.zone_low, scale)} – {fmt_price(r.zone_high, scale)} | "
                        f"Optimal Zone: {fmt_price(r.optimal_zone_low, scale)} – {fmt_price(r.optimal_zone_high, scale)}"
                    )
                st.caption(
                    f"Support: {fmt_price(r.nearest_support, scale)} | "
                    f"Resistance: {fmt_price(r.nearest_resistance, scale)} | "
                    f"{r.tp_obstacle_note}"
                )
                for n in r.mtf_notes:
                    st.caption(f"• {n}")

                if r.beyond_optimal_entry:
                    st.warning(f"Beyond Optimal Entry: {r.optimal_entry_status}")
                if r.bad_confluences:
                    st.caption("Confluence flags: " + ", ".join(r.bad_confluences))

                if is_confirmed:
                    if st.button(f"Lock [{r.timeframe}] {r.coin} into Active Trades", key=f"lock_{r.coin}_{r.timeframe}_{r.pattern_type}_{idx}"):
                        already = any(
                            t["coin"] == r.coin and t["timeframe"] == r.timeframe and t["pattern_type"] == r.pattern_type
                            for t in st.session_state.active_trades
                        )
                        if not already:
                            st.session_state.active_trades.append(
                                {
                                    "coin": r.coin,
                                    "timeframe": r.timeframe,
                                    "pattern_type": r.pattern_type,
                                    "sig_code": r.sig_code,
                                    "conf_score": r.conf_score,
                                    "conf_pct": r.conf_pct,
                                    "total_score": r.total_score,
                                    "bad_confluences": r.bad_confluences,
                                    "beyond_optimal_entry": r.beyond_optimal_entry,
                                    "optimal_entry_status": r.optimal_entry_status,
                                    "entry_d": r.entry_d,
                                    "sl": r.sl,
                                    "suggested_sl": r.suggested_sl,
                                    "bias_change_level": r.bias_change_level,
                                    "tp1": r.tp1,
                                    "tp2": r.tp2,
                                    "tp3": r.tp3,
                                    "tp1_hit": r.tp1_hit,
                                    "tp2_hit": r.tp2_hit,
                                    "tp3_hit": False,
                                    "sl_hit": False,
                                    "mtf_notes": r.mtf_notes,
                                    "tp_obstacle_note": r.tp_obstacle_note,
                                    "last_price": corr_by_coin[r.coin]["kucoin"] or r.last_close,
                                }
                            )
                            st.rerun()

    with col_ready:
        render_signal_panel("Ready to Be Panel", ready_items, is_confirmed=False)

    with col_conf:
        render_signal_panel("Confirmed Panel", conf_items, is_confirmed=True)

with pattern_tabs[1]:
    st.info("Additional harmonic patterns will appear in their own tabs here as we add them.")

# 3. STACKED CHARTS (TradingView up, Harmonic/Zone/Wall chart down) + Bullet Data & AI Summary
st.divider()
st.subheader("Chart & Setup Inspector")

ctl1, ctl2, ctl3 = st.columns([1.4, 1.0, 1.6])
with ctl1:
    chart_coin = st.selectbox("Coin", options=list(coins_tuple), index=0)
with ctl2:
    chart_tf = st.selectbox("Timeframe", options=selected_tfs, index=0)
with ctl3:
    corr_info = corr_by_coin[chart_coin]
    liq_info = fetch_liquidation_snapshot(chart_coin, corr_info.get("kucoin"))
    diff_str = f"{corr_info['diff_pct']:+.2f}%" if corr_info["diff_pct"] is not None else "-"
    st.metric(
        label=f"{chart_coin} KuCoin vs Web Ref",
        value=fmt_price(corr_info["kucoin"]),
        delta=f"Web {fmt_price(corr_info['web'])} ({diff_str}) · {liq_info['liq_note']}",
    )

selected_scan = result_map.get((chart_coin, chart_tf))
selected_df = df_lookup.get((chart_coin, chart_tf), pd.DataFrame())
chart_scale = 1.0
if price_mode == "Web / Reference Adjusted" and corr_info["ratio"]:
    chart_scale = float(corr_info["ratio"])

st.markdown("**1. Live TradingView Chart**")
tv_sym = f"KUCOIN:{to_kucoin_symbol(chart_coin).replace('-', '')}"
render_tradingview_chart(tv_sym, chart_tf, height=430)

st.markdown("**2. Harmonic Legs, Reversal Zone, Support/Resistance & Liquidation Chart**")
if selected_scan:
    st.plotly_chart(
        render_pattern_mapping_chart(selected_df, selected_scan, liq_info, chart_scale),
        use_container_width=True,
    )

# Bullet-point readout & AI Layman Summary
detail_tabs = st.tabs(["Data in Bullet Points", "AI Layman Opinion"])

with detail_tabs[0]:
    if selected_scan:
        bullets = [
            f"Coin & Timeframe: {selected_scan.coin}/USDT on {selected_scan.timeframe} ({selected_scan.pattern_type or 'NONE'} — {selected_scan.status_label})",
            f"Signal & Confluence: {selected_scan.sig_code} | Confluence {selected_scan.conf_pct:.1f}% ({selected_scan.conf_score}/3) | Total Score {selected_scan.total_score}/100",
            f"Entry (D): {fmt_price(selected_scan.entry_d, chart_scale)} | Optimal Zone: {fmt_price(selected_scan.optimal_zone_low, chart_scale)} – {fmt_price(selected_scan.optimal_zone_high, chart_scale)} ({selected_scan.optimal_entry_status})",
            f"Risk Levels: Primary SL {fmt_price(selected_scan.sl, chart_scale)} | Wick-Buffer SL {fmt_price(selected_scan.suggested_sl, chart_scale)} | Bias-Change {fmt_price(selected_scan.bias_change_level, chart_scale)}",
            f"Targets: TP1 {fmt_price(selected_scan.tp1, chart_scale)} | TP2 {fmt_price(selected_scan.tp2, chart_scale)} | TP3 {fmt_price(selected_scan.tp3, chart_scale)}",
            f"Walls & Obstacles: Support {fmt_price(selected_scan.nearest_support, chart_scale)} | Resistance {fmt_price(selected_scan.nearest_resistance, chart_scale)} | {selected_scan.tp_obstacle_note}",
            f"Score Adjustments: {', '.join(selected_scan.score_adjustments) if selected_scan.score_adjustments else 'None'}",
        ] + [f"MTF Note: {m}" for m in selected_scan.mtf_notes]
        for b in bullets:
            st.markdown(f"* {b}")

with detail_tabs[1]:
    if selected_scan:
        summary_key = f"{selected_scan.coin}_{selected_scan.timeframe}_{selected_scan.pattern_type}_{selected_scan.status_label}_{selected_scan.total_score}"
        if st.button("Generate Layman Summary", key="btn_layman_ai"):
            payload = {
                "coin": selected_scan.coin,
                "timeframe": selected_scan.timeframe,
                "pattern_type": selected_scan.pattern_type,
                "status_label": selected_scan.status_label,
                "conf_pct": selected_scan.conf_pct,
                "total_score": selected_scan.total_score,
                "entry_d": selected_scan.entry_d,
                "sl": selected_scan.sl,
                "suggested_sl": selected_scan.suggested_sl,
                "bias_change_level": selected_scan.bias_change_level,
                "tp_obstacle_note": selected_scan.tp_obstacle_note,
                "notes": selected_scan.mtf_notes,
            }
            st.session_state.ai_summaries[summary_key] = generate_layman_ai_summary(payload, ai_api_key)

        if summary_key in st.session_state.ai_summaries:
            st.info(st.session_state.ai_summaries[summary_key])
        else:
            st.caption("Click **Generate Layman Summary** to get a plain-language trade opinion (uses your AI key if entered in the sidebar).")

# 4. STATUS TABLE & SIGNAL AUDIT LOGBOOK
st.divider()
st.subheader("Status Table (All Coins × Timeframes)")

status_records = []
for sym in symbols_list:
    coin = clean_coin_name(sym)
    rec: Dict[str, str] = {"Coin": coin}
    for tf in selected_tfs:
        match = result_map.get((coin, tf))
        if not match or match.status_label == "IDLE":
            rec[tf] = "Idle (-)"
        elif match.status_label == "READY TO BE":
            rec[tf] = f"Ready to Be ({match.sig_code} {match.pattern_type[:4]} · {match.total_score})"
        else:
            rec[tf] = f"Confirmed ({match.sig_code} {match.pattern_type[:4]} · {match.total_score})"
    status_records.append(rec)

st.dataframe(pd.DataFrame(status_records), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Core Signal Audit Logbook")
log_tabs = st.tabs(HORIZONS)
for tab, horizon in zip(log_tabs, HORIZONS):
    with tab:
        scoped = filter_logbook_records(st.session_state.logbook, horizon)
        total_logged = len(scoped)
        tp_hits = sum(1 for r in scoped if str(r.get("outcome", "")).startswith("TP"))
        sl_hits = sum(1 for r in scoped if "SL" in str(r.get("outcome", "")).upper())
        still_tracking = max(0, total_logged - tp_hits - sl_hits)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals Logged", total_logged)
        m2.metric("TP Hits (TP1+)", tp_hits)
        m3.metric("SL Hits", sl_hits)
        m4.metric("Still Tracking", still_tracking)

        if scoped:
            for i, r in enumerate(scoped[:30], start=1):
                st.markdown(
                    f"**#{i} | {r.get('coin')}USDT [{r.get('timeframe')}] {r.get('pattern_type')}** — "
                    f"Outcome: `{r.get('outcome', 'TRACKING')}` · Stage: `{r.get('status_label')}` · "
                    f"Score: `{r.get('total_score', 0)}/100` · Logged: `{r.get('timestamp_pkt')}`  \n"
                    f"Entry: `{fmt_price(r.get('entry_d'))}` | SL: `{fmt_price(r.get('sl'))}` | "
                    f"TP1: `{fmt_price(r.get('tp1'))}` | TP2: `{fmt_price(r.get('tp2'))}` | TP3: `{fmt_price(r.get('tp3'))}`"
                )
        else:
            st.info("No logged signals for this window yet.")

if auto_refresh:
    time.sleep(int(refresh_sec))
    st.rerun()
