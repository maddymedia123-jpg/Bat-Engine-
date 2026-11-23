from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from bat_engine import BatScanResult, run_bat_engine
from logbook import HORIZONS, filter_logbook_by_horizon, load_logbook, record_generated_signals
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
    to_kucoin_symbol,
)

st.set_page_config(page_title="Harmonics Engine — KuCoin Bat Scanner", layout="wide")


def fmt_price(val: Optional[float], scale: float = 1.0) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    adj = val * scale
    return f"{adj:.6f}".rstrip("0").rstrip(".")


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
        pct = ((live_price - entry) / (tp - entry)) * 100.0
        if sl is not None and live_price <= sl:
            return 0.0
    else:
        pct = ((entry - live_price) / (entry - tp)) * 100.0
        if sl is not None and live_price >= sl:
            return 0.0
    return float(max(0.0, min(100.0, pct)))


def render_tradingview_chart(tv_symbol: str, tf_label: str, height: int = 380) -> None:
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


def render_pattern_mapping_chart(df: pd.DataFrame, res: BatScanResult, scale: float = 1.0) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
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
        formed_y = [
            v
            for i, v in zip((px_i, pa_i, pb_i, pc_i), (px_v, pa_v, pb_v, pc_v))
            if i >= 0
        ]
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
            fillcolor="rgba(0, 200, 83, 0.14)" if res.pattern_type == "BULLISH" else "rgba(255, 59, 59, 0.14)",
            line_width=1,
            line_dash="dot",
            annotation_text=res.zone_label,
            annotation_position="top left",
        )

    if res.entry_d is not None:
        fig.add_hline(y=res.entry_d * scale, line_dash="dash", line_color="#FFD54F", annotation_text="Entry (D)")
    if res.sl is not None:
        fig.add_hline(y=res.sl * scale, line_dash="solid", line_color="#FF3B3B", annotation_text="SL")
    for tp_val, label in ((res.tp1, "TP1"), (res.tp2, "TP2"), (res.tp3, "TP3")):
        if tp_val is not None:
            fig.add_hline(y=tp_val * scale, line_dash="dot", line_color="#3DDC84", annotation_text=label)

    fig.update_layout(
        template="plotly_dark",
        height=380,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# Session state initialization
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(DEFAULT_20_SYMBOLS)
if "active_trades" not in st.session_state:
    st.session_state.active_trades = []
if "logbook" not in st.session_state:
    st.session_state.logbook = load_logbook()

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
    st.subheader("Price & Display Options")
    price_mode = st.radio(
        "Display levels in:",
        options=["KuCoin Native", "Web / Reference Adjusted"],
        index=0,
    )
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

st.title("Harmonics Engine — Multi-Timeframe Bat Scanner")
st.caption(
    "Always-on monitoring across 1H, 2H, 4H & 1D with READY TO BE and CONFIRMED signal terminals, "
    "always-visible Active Trade Monitoring (PKT), reversal zones, chart mapping, and backtest logbook."
)

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

# Log all generated signals to core logbook automatically
st.session_state.logbook = record_generated_signals(st.session_state.logbook, results)

# Update active locked trades live (TP hits, SL hit after TP or standalone SL invalidation, confluence health)
result_map: Dict[tuple, BatScanResult] = {(r.coin, r.timeframe): r for r in results}
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
        trade["bad_confluences"] = latest_scan.bad_confluences
        trade["beyond_optimal_entry"] = latest_scan.beyond_optimal_entry
        trade["optimal_entry_status"] = latest_scan.optimal_entry_status

    if live_price is not None:
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

# 1. ACTIVE TRADE MONITORING TERMINAL (Always visible in background)
st.subheader("Active Trade Monitoring Terminal (PKT)")
if st.session_state.active_trades:
    for idx, trade in enumerate(list(st.session_state.active_trades)):
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

        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([2.2, 2.0, 2.5, 0.9])
            with h1:
                st.markdown(f"**{coin}/USDT · {direction} · [{tf}]** ({trade.get('sig_code', 'C')})")
                st.caption(
                    f"Live: **{fmt_price(live_p)}** | Entry: **{fmt_price(entry_p)}** | SL: **{fmt_price(sl_p)}**"
                )
            with h2:
                st.markdown(f"**Confluence: {trade.get('conf_pct', 0.0):.1f}%** ({trade.get('conf_score', 0)}/3)")
                if trade.get("beyond_optimal_entry"):
                    st.warning(f"Beyond Optimal Entry: {trade.get('optimal_entry_status')}")
                if trade.get("bad_confluences"):
                    st.error("Deteriorating Confluence: " + ", ".join(trade["bad_confluences"]))
                else:
                    st.success("Confluences Healthy")

            with h3:
                if sl_hit and not any_tp_hit:
                    st.error("SL INVALIDATION HIT — TP status bars cleared.")
                else:
                    if sl_hit and any_tp_hit:
                        tp_done = [
                            t for t, flag in (("TP1", trade.get("tp1_hit")), ("TP2", trade.get("tp2_hit")), ("TP3", trade.get("tp3_hit"))) if flag
                        ]
                        st.warning(f"SL HIT AFTER {', '.join(tp_done)} — Keeping TP status & SL hit until killed.")

                    for tp_label, tp_val, hit_flag, mult_h in (
                        ("TP1", tp1_p, trade.get("tp1_hit", False), 4.0 * tf_h),
                        ("TP2", tp2_p, trade.get("tp2_hit", False), 8.0 * tf_h),
                        ("TP3", tp3_p, trade.get("tp3_hit", False), 16.0 * tf_h),
                    ):
                        prog = 100.0 if hit_flag else compute_tp_progress(live_p, entry_p, tp_val, sl_p, direction)
                        eta_str = "HIT" if hit_flag else format_eta_pkt(mult_h * (1.0 - prog / 100.0))
                        st.progress(int(round(prog)), text=f"{tp_label} ({fmt_price(tp_val)}): {prog:.0f}% | {eta_str}")

            with h4:
                if st.button("Kill", key=f"kill_trade_{idx}_{coin}_{tf}", use_container_width=True):
                    st.session_state.active_trades.pop(idx)
                    st.rerun()
else:
    st.info("No active trades locked yet. Lock any CONFIRMED signal below to monitor it live here.")

# 2. READY TO BE & CONFIRMED SIGNAL TERMINALS
st.divider()
col_ready, col_conf = st.columns(2)

ready_items = [
    r for r in results if r.status_label == "READY TO BE" and (show_overstretched or not r.overstretched)
]
conf_items = [r for r in results if r.status_label == "CONFIRMED"]


def render_signal_terminal(title: str, items: List[BatScanResult], is_confirmed: bool) -> None:
    st.subheader(title)
    if not items:
        st.info(f"No {title.lower()} signals right now.")
        return

    for idx, r in enumerate(items):
        scale = 1.0
        if price_mode == "Web / Reference Adjusted" and corr_by_coin[r.coin]["ratio"]:
            scale = float(corr_by_coin[r.coin]["ratio"])

        with st.container(border=True):
            c_top1, c_top2 = st.columns([3, 1])
            with c_top1:
                st.markdown(
                    f"**{r.coin}/USDT · {r.pattern_type} · [{r.timeframe}]** — `{r.sig_code}` ({r.conf_pct:.1f}%)"
                )
            with c_top2:
                st.caption(r.status_label)

            st.caption(
                f"Entry (D): **{fmt_price(r.entry_d, scale)}** | SL: **{fmt_price(r.sl, scale)}** | "
                f"TP1: **{fmt_price(r.tp1, scale)}** | TP2/3: **{fmt_price(r.tp2, scale)} / {fmt_price(r.tp3, scale)}**"
            )
            if r.zone_low is not None and r.zone_high is not None:
                st.caption(f"{r.zone_label}: **{fmt_price(r.zone_low, scale)} – {fmt_price(r.zone_high, scale)}**")

            if r.beyond_optimal_entry:
                st.warning(f"Beyond Optimal Entry: {r.optimal_entry_status}")
            if r.bad_confluences:
                st.caption("Confluence flags: " + ", ".join(r.bad_confluences))

            if is_confirmed:
                lock_key = f"lock_{r.coin}_{r.timeframe}_{r.pattern_type}_{idx}"
                if st.button(f"Lock [{r.timeframe}] {r.coin} into Active Trades", key=lock_key):
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
                                "bad_confluences": r.bad_confluences,
                                "beyond_optimal_entry": r.beyond_optimal_entry,
                                "optimal_entry_status": r.optimal_entry_status,
                                "entry_d": r.entry_d,
                                "sl": r.sl,
                                "tp1": r.tp1,
                                "tp2": r.tp2,
                                "tp3": r.tp3,
                                "tp1_hit": r.tp1_hit,
                                "tp2_hit": r.tp2_hit,
                                "tp3_hit": False,
                                "sl_hit": False,
                                "last_price": corr_by_coin[r.coin]["kucoin"] or r.last_close,
                            }
                        )
                        st.rerun()


with col_ready:
    render_signal_terminal("Ready to Be (Forming / Expected D)", ready_items, is_confirmed=False)

with col_conf:
    render_signal_terminal("Confirmed (D Hit)", conf_items, is_confirmed=True)

# 3. CHARTING & CORRELATION PANEL
st.divider()
st.subheader("Harmonic Pattern Mapping, Live TradingView & KuCoin ↔ Web Correlation")

sel_c1, sel_c2, sel_c3 = st.columns([1.5, 1.0, 1.5])
with sel_c1:
    chart_coin = st.selectbox("Select Coin for Chart", options=list(coins_tuple), index=0)
with sel_c2:
    chart_tf = st.selectbox("Select Timeframe for Chart", options=selected_tfs, index=0)
with sel_c3:
    corr_info = corr_by_coin[chart_coin]
    liq_info = fetch_liquidation_snapshot(chart_coin)
    diff_str = f"{corr_info['diff_pct']:+.2f}%" if corr_info["diff_pct"] is not None else "-"
    st.metric(
        label=f"{chart_coin} KuCoin vs Web Ref",
        value=fmt_price(corr_info["kucoin"]),
        delta=f"Web {fmt_price(corr_info['web'])} ({diff_str}) · Liq: {liq_info['liq_bias']}",
    )

selected_scan = result_map.get((chart_coin, chart_tf))
selected_df = df_lookup.get((chart_coin, chart_tf), pd.DataFrame())
chart_scale = 1.0
if price_mode == "Web / Reference Adjusted" and corr_info["ratio"]:
    chart_scale = float(corr_info["ratio"])

map_col, tv_col, card_col = st.columns([1.45, 1.45, 0.9])
with map_col:
    st.caption("Harmonic Leg & Reversal Zone Map")
    if selected_scan:
        st.plotly_chart(render_pattern_mapping_chart(selected_df, selected_scan, chart_scale), use_container_width=True)

with tv_col:
    st.caption("Live TradingView Chart")
    tv_sym = f"KUCOIN:{to_kucoin_symbol(chart_coin).replace('-', '')}"
    render_tradingview_chart(tv_sym, chart_tf, height=380)

with card_col:
    st.caption("Bat Engine Status Card")
    if selected_scan:
        card_rows = [
            {"Field": "Pattern Action", "Value": f"{selected_scan.pattern_type} {selected_scan.status_label}".strip()},
            {"Field": "Signal / Conf", "Value": f"{selected_scan.sig_code} ({selected_scan.conf_pct:.1f}%)"},
            {"Field": "Leg X -> A", "Value": selected_scan.leg_statuses.get("Leg X -> A", "-")},
            {"Field": "Leg A -> B", "Value": selected_scan.leg_statuses.get("Leg A -> B", "-")},
            {"Field": "Leg B -> C", "Value": selected_scan.leg_statuses.get("Leg B -> C", "-")},
            {"Field": "Leg C -> D", "Value": selected_scan.leg_statuses.get("Leg C -> D", "-")},
            {"Field": "Entry (D)", "Value": fmt_price(selected_scan.entry_d, chart_scale)},
            {"Field": "SL (ATR Buf)", "Value": fmt_price(selected_scan.sl, chart_scale)},
            {"Field": "TP1 / TP2 / TP3", "Value": f"{fmt_price(selected_scan.tp1, chart_scale)} / {fmt_price(selected_scan.tp2, chart_scale)} / {fmt_price(selected_scan.tp3, chart_scale)}"},
            {"Field": "Zone", "Value": selected_scan.zone_label},
        ]
        st.dataframe(pd.DataFrame(card_rows), use_container_width=True, hide_index=True, height=380)

# 4. STATUS TABLE & 1D / 4H / 1W / 1M LOGBOOK
st.divider()
st.subheader("Status Table (All Watchlist Coins × Timeframes)")

status_records = []
for sym in symbols_list:
    coin = clean_coin_name(sym)
    rec: Dict[str, str] = {"Coin": coin}
    for tf in selected_tfs:
        match = result_map.get((coin, tf))
        if not match or match.status_label == "IDLE":
            rec[tf] = "Idle (-)"
        elif match.status_label == "READY TO BE":
            rec[tf] = f"Ready to Be ({match.sig_code} {match.pattern_type[:4]})"
        else:
            rec[tf] = f"Confirmed ({match.sig_code} {match.pattern_type[:4]})"
    status_records.append(rec)

st.dataframe(pd.DataFrame(status_records), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Core Signal Logbook (1D / 4H / 1W / 1M Backtest Record)")
log_tabs = st.tabs(HORIZONS)
for tab, horizon in zip(log_tabs, HORIZONS):
    with tab:
        st.dataframe(filter_logbook_by_horizon(st.session_state.logbook, horizon), use_container_width=True, hide_index=True)

# Continuous background refresh loop for live monitoring
if auto_refresh:
    import time

    time.sleep(int(refresh_sec))
    st.rerun()
