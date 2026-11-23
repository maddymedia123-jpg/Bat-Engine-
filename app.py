from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from bat_engine import BatScanResult, enrich_multi_timeframe_context, run_bat_engine
from market_data import (
    DEFAULT_PERP_SYMBOLS,
    TF_HOURS,
    TIMEFRAME_OPTIONS,
    TV_INTERVAL_MAP,
    clean_coin_name,
    fetch_liquidation_snapshot,
    fetch_perp_klines,
    fetch_perp_last_price,
    format_eta_pkt,
    to_perp_symbol,
)
from turso_store import (
    HORIZONS,
    filter_logbook_records,
    init_storage,
    load_active_trades,
    load_audit_logs,
    load_logbook,
    load_saved_summaries,
    record_generated_signals,
    run_on_demand_trader_agent,
    save_active_trades,
    save_summary,
    silent_self_audit_and_update,
    turso_configured,
)

st.set_page_config(page_title="Harmonics Engine — Bat Pattern", layout="wide")


def fmt_price(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return f"${val:.6f}".rstrip("0").rstrip(".")


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
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name=f"{res.coin} {res.timeframe}",
        )
    )

    if res.pivots and all(k in res.pivots for k in ("X", "A", "B", "C")):
        px_i, px_v = res.pivots["X"][0] - offset, res.pivots["X"][1]
        pa_i, pa_v = res.pivots["A"][0] - offset, res.pivots["A"][1]
        pb_i, pb_v = res.pivots["B"][0] - offset, res.pivots["B"][1]
        pc_i, pc_v = res.pivots["C"][0] - offset, res.pivots["C"][1]

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
                    y=[pc_v, res.entry_d],
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
            y0=res.zone_low,
            y1=res.zone_high,
            fillcolor="rgba(0, 200, 83, 0.15)" if res.pattern_type == "BULLISH" else "rgba(255, 59, 59, 0.15)",
            line_width=1,
            line_dash="dot",
            annotation_text=res.zone_label,
            annotation_position="top left",
        )

    if res.entry_d is not None:
        fig.add_hline(y=res.entry_d, line_dash="dash", line_color="#FFD54F", annotation_text="Entry (D)")
    if res.sl is not None:
        fig.add_hline(y=res.sl, line_dash="solid", line_color="#FF3B3B", annotation_text="ATR SL")
    if res.secondary_sl is not None:
        fig.add_hline(y=res.secondary_sl, line_dash="dashdot", line_color="#FF9800", annotation_text="Secondary SL")
    if res.bias_change_level is not None:
        fig.add_hline(y=res.bias_change_level, line_dash="longdash", line_color="#AB47BC", annotation_text="Bias-Change")
    if res.nearest_support is not None:
        fig.add_hline(y=res.nearest_support, line_dash="dot", line_color="#29B6F6", annotation_text="Nearest Support")
    if res.nearest_resistance is not None:
        fig.add_hline(y=res.nearest_resistance, line_dash="dot", line_color="#FFA726", annotation_text="Nearest Resistance")

    upper_liq = liq_info.get("upper_liq")
    lower_liq = liq_info.get("lower_liq")
    if isinstance(upper_liq, (int, float)):
        fig.add_hline(y=float(upper_liq), line_dash="dot", line_color="#EF5350", annotation_text="Est. Liq High")
    if isinstance(lower_liq, (int, float)):
        fig.add_hline(y=float(lower_liq), line_dash="dot", line_color="#26A69A", annotation_text="Est. Liq Low")

    for tp_val, label in ((res.tp1, "TP1"), (res.tp2, "TP2"), (res.tp3, "TP3")):
        if tp_val is not None:
            fig.add_hline(y=tp_val, line_dash="dot", line_color="#3DDC84", annotation_text=label)

    fig.update_layout(
        template="plotly_dark",
        height=430,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


if "storage_ready" not in st.session_state:
    init_storage()
    st.session_state.storage_ready = True
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(DEFAULT_PERP_SYMBOLS)
if "active_trades" not in st.session_state:
    st.session_state.active_trades = load_active_trades()
if "logbook" not in st.session_state:
    st.session_state.logbook = load_logbook()
if "ai_summaries" not in st.session_state:
    st.session_state.ai_summaries = load_saved_summaries()
if "audit_logs" not in st.session_state:
    st.session_state.audit_logs = load_audit_logs()
if "show_summary_toggle" not in st.session_state:
    st.session_state.show_summary_toggle = True

with st.sidebar:
    st.header("Perp Watchlist & Timeframes")
    st.caption(f"Turso Cloud: {'Connected' if turso_configured() else 'Local Fallback (Add Secrets for Turso)'}")

    add_col, del_col = st.columns(2)
    with add_col:
        new_coin_raw = st.text_input("Add Perp", placeholder="e.g. BTCUSDT.P or SOL")
        if st.button("Add Coin", use_container_width=True) and new_coin_raw.strip():
            formatted = to_perp_symbol(new_coin_raw.strip())
            if formatted not in st.session_state.watchlist:
                st.session_state.watchlist.append(formatted)
                st.rerun()

    with del_col:
        del_target = st.selectbox("Delete Coin", options=["-"] + st.session_state.watchlist)
        if st.button("Delete Coin", use_container_width=True) and del_target != "-":
            st.session_state.watchlist = [c for c in st.session_state.watchlist if c != del_target]
            st.rerun()

    st.caption(f"Active Watchlist Count: {len(st.session_state.watchlist)} perps")
    selected_tfs = st.multiselect(
        "Timeframes (1H, 2H, 4H, 1D)",
        options=list(TIMEFRAME_OPTIONS.keys()),
        default=["1H", "2H", "4H", "1D"],
    )

    st.divider()
    st.subheader("Live Scanning")
    auto_refresh = st.checkbox("Smooth auto-refresh", value=True)
    refresh_sec = st.slider("Refresh interval (sec)", min_value=20, max_value=180, value=60, step=10)

    st.divider()
    st.subheader("AI Agent Key (On-Demand Only)")
    ai_api_key = st.text_input("AI API Key Override (Optional)", type="password", placeholder="sk-...")

    st.divider()
    st.subheader("Display Options")
    show_overstretched = st.checkbox("Show overstretched READY TO BE (W) rows", value=False)

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

symbols_list = st.session_state.watchlist if st.session_state.watchlist else list(DEFAULT_PERP_SYMBOLS)
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
st.caption("Pine v6 BatPattern engine on perpetual futures (defaulting to BTCUSDT.P) with ATR SL, Turso persistence, and on-demand AI agent.")

coins_list = [clean_coin_name(s) for s in symbols_list]
last_price_by_coin: Dict[str, Optional[float]] = {}
for sym in symbols_list:
    coin = clean_coin_name(sym)
    last_price_by_coin[coin] = fetch_perp_last_price(sym)

results: List[BatScanResult] = []
df_lookup: Dict[tuple, pd.DataFrame] = {}

for tf in selected_tfs:
    for sym in symbols_list:
        coin = clean_coin_name(sym)
        try:
            df = fetch_perp_klines(sym, tf, limit=250)
            if last_price_by_coin.get(coin) is None and not df.empty:
                last_price_by_coin[coin] = float(df["close"].iloc[-1])
        except Exception as exc:
            st.warning(f"Perp fetch issue for {coin} ({tf}): {exc}")
            df = pd.DataFrame()
        df_lookup[(coin, tf)] = df
        results.append(run_bat_engine(df, sym, coin, tf, params))

results = enrich_multi_timeframe_context(results)
st.session_state.logbook = record_generated_signals(st.session_state.logbook, results)
result_map: Dict[tuple, BatScanResult] = {(r.coin, r.timeframe): r for r in results}

trades_changed = False
for trade in st.session_state.active_trades:
    coin = trade["coin"]
    tf = trade["timeframe"]
    direction = trade["pattern_type"]
    live_price = last_price_by_coin.get(coin)
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
        trade["secondary_sl"] = latest_scan.secondary_sl
        trade["bias_change_level"] = latest_scan.bias_change_level
        trade["mtf_notes"] = latest_scan.mtf_notes
        trade["tp_obstacle_note"] = latest_scan.tp_obstacle_note

    if live_price is not None and not trade.get("sl_hit", False):
