from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from bat_engine import BatScanResult, enrich_multi_timeframe_context, run_bat_engine
from market_data import (
    DEFAULT_PERP_SYMBOLS,
    TF_HOURS,
    TF_ORDER,
    TIMEFRAME_OPTIONS,
    TV_INTERVAL_MAP,
    clean_coin_name,
    compute_price_correlation,
    fetch_liquidation_snapshot,
    fetch_perp_klines,
    fetch_perp_last_price,
    fetch_web_reference_prices,
    format_eta_pkt,
    to_perp_symbol,
)
from turso_store import (
    HORIZONS,
    PKT_TZ,
    audit_and_correlate_summary,
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
    send_discord_webhook_alert,
    sync_logbook_outcomes,
    turso_configured,
)

st.set_page_config(page_title="Harmonics Engine — Bat Pattern", layout="wide")

GATE_INTERVAL_MAP: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
    "1D": "1d",
}


def to_gate_contract(coin_or_sym: str) -> str:
    coin = clean_coin_name(coin_or_sym).upper()
    return f"{coin}_USDT"


@st.cache_data(ttl=20)
def fetch_gate_last_price(coin_or_sym: str) -> Optional[float]:
    contract = to_gate_contract(coin_or_sym)
    try:
        resp = requests.get(
            "https://api.gateio.ws/api/v4/futures/usdt/tickers",
            params={"contract": contract},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            val = data[0].get("last") or data[0].get("mark_price")
            if val is not None:
                return float(val)
    except Exception:
        pass
    return None


@st.cache_data(ttl=30)
def fetch_gate_klines(coin_or_sym: str, tf_label: str, limit: int = 250) -> pd.DataFrame:
    contract = to_gate_contract(coin_or_sym)
    interval = GATE_INTERVAL_MAP.get(tf_label, "1h")
    try:
        resp = requests.get(
            "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
            params={"contract": contract, "interval": interval, "limit": min(limit, 500)},
            timeout=8,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()
        recs = []
        for r in rows:
            if isinstance(r, dict):
                recs.append(
                    {
                        "open": float(r.get("o", 0.0)),
                        "high": float(r.get("h", 0.0)),
                        "low": float(r.get("l", 0.0)),
                        "close": float(r.get("c", 0.0)),
                        "volume": float(r.get("v", r.get("sum", 0.0)) or 0.0),
                    }
                )
        return pd.DataFrame(recs)
    except Exception:
        return pd.DataFrame()


def fmt_price(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return f"${val:.6f}".rstrip("0").rstrip(".")


def format_tf_counts(items: List[BatScanResult], tfs: List[str]) -> str:
    ordered_tfs = [tf for tf in TF_ORDER if tf in tfs] + [tf for tf in tfs if tf not in TF_ORDER]
    parts: List[str] = []
    for tf in ordered_tfs:
        count = sum(1 for item in items if item.timeframe == tf)
        parts.append(f"{tf} ({count})")
    return " · ".join(parts)


def evaluate_sl_and_tp_posture(trade: Dict[str, object]) -> Tuple[str, Optional[float], str]:
    direction = str(trade.get("pattern_type") or "BULLISH")
    live_p = trade.get("last_price")
    entry_p = trade.get("entry_d")
    primary_sl = trade.get("sl")
    secondary_sl = trade.get("secondary_sl")
    be_sl = trade.get("break_even_sl") or entry_p
    tp1_hit = bool(trade.get("tp1_hit", False))
    tp2_hit = bool(trade.get("tp2_hit", False))
    tp3_hit = bool(trade.get("tp3_hit", False))
    sl_hit = bool(trade.get("sl_hit", False))

    if sl_hit:
        return "SL HIT", primary_sl if isinstance(primary_sl, (int, float)) else None, "SL hit—trade closing after grace window."
    if tp1_hit and tp2_hit and tp3_hit:
        return "TP HIT", None, "All TPs hit—trade closing after grace window."
    if tp2_hit:
        be_val = float(be_sl) if isinstance(be_sl, (int, float)) else (float(entry_p) if isinstance(entry_p, (int, float)) else None)
        return "BREAK-EVEN SL (TP3 FOCUS)", be_val, "TP2 hit—keep SL at Break-Even and target TP3."
    if tp1_hit:
        be_val = float(be_sl) if isinstance(be_sl, (int, float)) else (float(entry_p) if isinstance(entry_p, (int, float)) else None)
        return (
            "SHIFT SL TO BREAK-EVEN -> LOOK FOR TP2",
            be_val,
            "TP1 hit—close here or shift SL to Break-Even for TP2.",
        )

    fav_move = False
    if isinstance(live_p, (int, float)) and isinstance(entry_p, (int, float)):
        fav_move = (direction == "BULLISH" and live_p > entry_p) or (direction == "BEARISH" and live_p < entry_p)

    if fav_move and isinstance(secondary_sl, (int, float)):
        return "SECONDARY SL (TRAILING)", float(secondary_sl), "Price in favor—trail with Secondary SL."

    return "PRIMARY SL", float(primary_sl) if isinstance(primary_sl, (int, float)) else None, "Hold Primary SL."


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
    is_committed: bool = False,
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

    if not is_committed and res.zone_low is not None and res.zone_high is not None:
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
        fig.add_hline(
            y=res.entry_d,
            line_dash="dash",
            line_color="#FFD54F",
            annotation_text="Committed Entry" if is_committed else "Optimal Entry (D)",
        )
    if res.sl is not None:
        fig.add_hline(y=res.sl, line_dash="solid", line_color="#FF3B3B", annotation_text="Primary ATR SL")
    if res.secondary_sl is not None:
        fig.add_hline(y=res.secondary_sl, line_dash="dashdot", line_color="#FF9800", annotation_text="Secondary SL")
    if res.break_even_sl is not None and is_committed:
        fig.add_hline(y=res.break_even_sl, line_dash="longdash", line_color="#00E5FF", annotation_text="Break-Even SL")
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
    st.header("Swap Futures Watchlist & Timeframes")
    st.caption(f"Turso Cloud: {'Connected' if turso_configured() else 'Local Fallback (Add Secrets for Turso)'}")

    add_col, del_col = st.columns(2)
    with add_col:
        new_coin_raw = st.text_input("Add Swap", placeholder="e.g. BTCUSDT.P or SOL")
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

    st.caption(f"Active Watchlist Count: {len(st.session_state.watchlist)} swap contracts")
    selected_tfs = st.multiselect(
        "Timeframes (1H, 2H, 4H, 1D)",
        options=list(TIMEFRAME_OPTIONS.keys()),
        default=["1H", "2H", "4H", "1D"],
    )

    st.divider()
    st.subheader("Exchange Feed (Binance vs Gate.io)")
    feed_mode = st.selectbox(
        "Data Source",
        options=["Auto (Closest to Binance)", "Gate.io Swap", "Binance Swap"],
        index=0,
    )
    post_exit_grace_hrs = st.slider(
        "Keep closed trade (TP/SL hit) before exit (hrs)",
        min_value=0.0,
        max_value=4.0,
        value=2.0,
        step=0.5,
    )

    st.divider()
    st.subheader("Live Scanning (Non-Blocking)")
    auto_refresh = st.checkbox("Auto-update active monitor fragment", value=False)
    refresh_sec = st.slider("Fragment update interval (sec)", min_value=15, max_value=180, value=30, step=5)

    st.divider()
    st.subheader("Discord Partner Alerts")
    discord_webhook_url = st.text_input(
        "Discord Webhook URL (or set DISCORD_WEBHOOK_URL in Secrets)",
        type="password",
        placeholder="https://discord.com/api/webhooks/...",
    ).strip()
    discord_enabled = st.checkbox("Enable Discord Bot Alerts", value=True)
    discord_low_threshold = st.slider("Discord Alert Probability Floor", min_value=25, max_value=80, value=45, step=5)

    st.divider()
    st.subheader("AI Agent Key (On-Demand Only)")
    ai_api_key_raw = st.text_input("AI API Key Override (Optional)", type="password", placeholder="sk-or-v1-...")
    ai_api_key = ai_api_key_raw.strip()

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
        max_signal_dist_pct = st.number_input("Max Signal Distance from Price (%)", min_value=2.0, max_value=100.0, value=18.0, step=1.0)

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
    "max_signal_dist_pct": float(max_signal_dist_pct),
}

st.title("Harmonics Engine")
st.caption("Pine v6 BatPattern engine on swap futures (Binance & Gate.io proximity check), auto-exit into backtest logbook on TP/SL hit, and master partner AI.")

coins_list = [clean_coin_name(s) for s in symbols_list]
last_price_by_coin: Dict[str, Optional[float]] = {}
source_used_by_coin: Dict[str, str] = {}
gate_diff_pct_by_coin: Dict[str, Optional[float]] = {}

for sym in symbols_list:
    coin = clean_coin_name(sym)
    b_price = fetch_perp_last_price(sym)
    g_price = fetch_gate_last_price(coin)
    diff_pct: Optional[float] = None
    if isinstance(b_price, (int, float)) and isinstance(g_price, (int, float)) and b_price > 0:
        diff_pct = abs(g_price - b_price) / b_price * 100.0
    gate_diff_pct_by_coin[coin] = diff_pct

    if feed_mode == "Gate.io Swap":
        last_price_by_coin[coin] = g_price if g_price is not None else b_price
        source_used_by_coin[coin] = "Gate.io" if g_price is not None else "Binance"
    elif feed_mode == "Binance Swap":
        last_price_by_coin[coin] = b_price if b_price is not None else g_price
        source_used_by_coin[coin] = "Binance" if b_price is not None else "Gate.io"
    else:
        if g_price is not None and (b_price is None or (diff_pct is not None and diff_pct <= 0.35)):
            last_price_by_coin[coin] = g_price
            source_used_by_coin[coin] = "Gate.io (Close)"
        else:
            last_price_by_coin[coin] = b_price if b_price is not None else g_price
            source_used_by_coin[coin] = "Binance"

results: List[BatScanResult] = []
df_lookup: Dict[tuple, pd.DataFrame] = {}

for tf in selected_tfs:
    for sym in symbols_list:
        coin = clean_coin_name(sym)
        try:
            b_df = fetch_perp_klines(sym, tf, limit=250)
            g_df = fetch_gate_klines(coin, tf, limit=250)
            if feed_mode == "Gate.io Swap":
                df = g_df if not g_df.empty else b_df
            elif feed_mode == "Binance Swap":
                df = b_df if not b_df.empty else g_df
            else:
                if not g_df.empty and not b_df.empty:
                    b_last = float(b_df["close"].iloc[-1])
                    g_last = float(g_df["close"].iloc[-1])
                    df = g_df if b_last > 0 and abs(g_last - b_last) / b_last <= 0.35 else b_df
                else:
                    df = g_df if not g_df.empty else b_df

            if last_price_by_coin.get(coin) is None and df is not None and not df.empty:
                last_price_by_coin[coin] = float(df["close"].iloc[-1])
        except Exception as exc:
            st.warning(f"Swap fetch issue for {coin} ({tf}): {exc}")
            df = pd.DataFrame()
        df_lookup[(coin, tf)] = df
        results.append(run_bat_engine(df, sym, coin, tf, params))

results = enrich_multi_timeframe_context(results)

# Deduplicated & strictly validated result list (deduped_results)
deduped_results: List[BatScanResult] = []
seen_pairs = set()
for r in results:
    expected_coin = clean_coin_name(r.symbol)
    if r.coin != expected_coin or r.timeframe not in selected_tfs:
        continue
    pair_key = (r.coin, r.timeframe)
    if pair_key in seen_pairs:
        continue
    seen_pairs.add(pair_key)
    deduped_results.append(r)

results = deduped_results
st.session_state.logbook = record_generated_signals(st.session_state.logbook, results)
result_map: Dict[tuple, BatScanResult] = {(r.coin, r.timeframe): r for r in results}

now_ts = datetime.now(PKT_TZ).timestamp()
trades_changed = False
surviving_trades: List[Dict[str, object]] = []

for trade in st.session_state.active_trades:
    coin = str(trade["coin"])
    tf = str(trade["timeframe"])
    direction = str(trade["pattern_type"])
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
        trade["momentum_note"] = latest_scan.momentum_note
        trade["beyond_optimal_entry"] = latest_scan.beyond_optimal_entry
        trade["optimal_entry_status"] = latest_scan.optimal_entry_status
        trade["optimal_zone_low"] = latest_scan.optimal_zone_low
        trade["optimal_zone_high"] = latest_scan.optimal_zone_high
        trade["secondary_sl"] = latest_scan.secondary_sl
        trade["break_even_sl"] = latest_scan.break_even_sl or trade.get("entry_d")
        trade["bias_change_level"] = latest_scan.bias_change_level
        trade["mtf_notes"] = latest_scan.mtf_notes
        trade["tp_obstacle_note"] = latest_scan.tp_obstacle_note

    if isinstance(live_price, (int, float)) and not trade.get("sl_hit", False):
        if direction == "BULLISH":
            if trade.get("tp1") is not None and live_price >= trade["tp1"] and not trade.get("tp1_hit"):
                trade["tp1_hit"] = True
                trades_changed = True
            if trade.get("tp2") is not None and live_price >= trade["tp2"] and not trade.get("tp2_hit"):
                trade["tp2_hit"] = True
                trades_changed = True
            if trade.get("tp3") is not None and live_price >= trade["tp3"] and not trade.get("tp3_hit"):
                trade["tp3_hit"] = True
                trades_changed = True
            if trade.get("sl") is not None and live_price <= trade["sl"]:
                trade["sl_hit"] = True
                trades_changed = True
        else:
            if trade.get("tp1") is not None and live_price <= trade["tp1"] and not trade.get("tp1_hit"):
                trade["tp1_hit"] = True
                trades_changed = True
            if trade.get("tp2") is not None and live_price <= trade["tp2"] and not trade.get("tp2_hit"):
                trade["tp2_hit"] = True
                trades_changed = True
            if trade.get("tp3") is not None and live_price <= trade["tp3"] and not trade.get("tp3_hit"):
                trade["tp3_hit"] = True
                trades_changed = True
            if trade.get("sl") is not None and live_price >= trade["sl"]:
                trade["sl_hit"] = True
                trades_changed = True

    all_tps_hit = bool(trade.get("tp1_hit") and trade.get("tp2_hit") and trade.get("tp3_hit"))
    sl_hit = bool(trade.get("sl_hit", False))
    is_finished = all_tps_hit or sl_hit

    if is_finished and not trade.get("closed_ts"):
        trade["closed_ts"] = now_ts
        trade["exit_reason"] = "TP HIT" if all_tps_hit else "SL HIT"
        trades_changed = True

    sl_mode, active_sl_val, sl_action_note = evaluate_sl_and_tp_posture(trade)
    trade["sl_mode"] = sl_mode
    trade["active_sl_guidance"] = active_sl_val
    trade["sl_action_note"] = sl_action_note

    if is_finished:
        elapsed_hrs = (now_ts - float(trade.get("closed_ts", now_ts))) / 3600.0
        if elapsed_hrs > float(post_exit_grace_hrs):
            trades_changed = True
            continue

    surviving_trades.append(trade)

if len(surviving_trades) != len(st.session_state.active_trades) or trades_changed:
    st.session_state.active_trades = surviving_trades
    save_active_trades(st.session_state.active_trades)

st.session_state.logbook = sync_logbook_outcomes(
    st.session_state.logbook,
    st.session_state.active_trades,
    last_price_by_coin,
)

committed_keys = {(t.get("coin"), t.get("timeframe")) for t in st.session_state.active_trades}


def _dispatch_discord_for_trade_state(trade: Dict[str, object], partner_summary: Optional[str] = None) -> None:
    if not discord_enabled:
        return
    score_val = int(trade.get("total_score", 0) or 0)
    coin = str(trade.get("coin", ""))
    tf = str(trade.get("timeframe", ""))
    direction = str(trade.get("pattern_type", ""))
    live_p = trade.get("last_price")
    entry_p = trade.get("entry_d")
    sl_mode = str(trade.get("sl_mode") or "PRIMARY SL")
    active_sl = trade.get("active_sl_guidance")
    tp1_hit = bool(trade.get("tp1_hit", False))
    tp2_hit = bool(trade.get("tp2_hit", False))
    tp3_hit = bool(trade.get("tp3_hit", False))
    sl_hit = bool(trade.get("sl_hit", False))
    wall = str(trade.get("tp_obstacle_note") or "Clear path")
    tf_h = TF_HOURS.get(tf, 1.0)
    eta_tp2 = str(trade.get("eta_tp2") or format_eta_pkt(8.0 * tf_h))
    eta_tp3 = str(trade.get("eta_tp3") or format_eta_pkt(16.0 * tf_h))

    if score_val < int(discord_low_threshold) and not (sl_hit or (tp1_hit and tp2_hit and tp3_hit)):
        return

    fields = [
        {"name": "Entry / Live", "value": f"{fmt_price(entry_p if isinstance(entry_p, (int, float)) else None)} -> {fmt_price(live_p if isinstance(live_p, (int, float)) else None)}", "inline": True},
        {"name": "Prob Score", "value": f"{score_val}/100 ({trade.get('conf_pct', 0):.0f}% Conf)", "inline": True},
        {"name": "SL Posture", "value": f"{sl_mode} ({fmt_price(active_sl if isinstance(active_sl, (int, float)) else None)})", "inline": True},
        {"name": "Targets & ETA", "value": f"TP1 {fmt_price(trade.get('tp1'))} | TP2 {fmt_price(trade.get('tp2'))} ({eta_tp2}) | TP3 {fmt_price(trade.get('tp3'))} ({eta_tp3})", "inline": False},
        {"name": "Partner Move", "value": str(trade.get("sl_action_note") or "-"), "inline": False},
    ]

    if partner_summary:
        send_discord_webhook_alert(
            event_key=f"partner_{coin}_{tf}_{sl_mode}_{int(tp1_hit)}{int(tp2_hit)}{int(tp3_hit)}",
            title=f"Partner Read — {coin}USDT.P [{tf}] ({direction})",
            description=partner_summary[:1800],
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0x3B82F6,
            force=True,
        )

    if sl_hit:
        send_discord_webhook_alert(
            event_key=f"sl_hit_{coin}_{tf}",
            title=f"SL HIT — {coin}USDT.P [{tf}] ({direction})",
            description=f"SL hit. Logged for backtest; closing active slot after {post_exit_grace_hrs}h window.",
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0xEF4444,
        )
    elif tp1_hit and tp2_hit and tp3_hit:
        send_discord_webhook_alert(
            event_key=f"all_tp_hit_{coin}_{tf}",
            title=f"TP HIT — {coin}USDT.P [{tf}] ({direction})",
            description=f"All TPs hit. Logged for backtest; closing active slot after {post_exit_grace_hrs}h window.",
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0x10B981,
        )
    elif tp2_hit:
        send_discord_webhook_alert(
            event_key=f"tp2_hit_{coin}_{tf}",
            title=f"TP2 Hit -> Break-Even & TP3 ({eta_tp3}) — {coin}USDT.P [{tf}]",
            description=f"TP2 hit. Hold Break-Even SL and watch TP3 ({fmt_price(trade.get('tp3'))}).",
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0x22C55E,
        )
    elif tp1_hit:
        send_discord_webhook_alert(
            event_key=f"tp1_hit_{coin}_{tf}",
            title=f"TP1 Hit -> Close or Shift SL to BE for TP2 ({eta_tp2}) — {coin}USDT.P [{tf}]",
            description=f"TP1 hit. Close at TP1 or shift SL to Break-Even and look for TP2 ({fmt_price(trade.get('tp2'))}).",
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0x00C853,
        )
    elif sl_mode == "SECONDARY SL (TRAILING)":
        send_discord_webhook_alert(
            event_key=f"sec_sl_{coin}_{tf}",
            title=f"SL Move -> Secondary SL — {coin}USDT.P [{tf}]",
            description=f"Price moved in favor; trail with Secondary SL ({fmt_price(active_sl if isinstance(active_sl, (int, float)) else None)}).",
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0xF59E0B,
        )

    if wall != "Clear path":
        send_discord_webhook_alert(
            event_key=f"wall_{coin}_{tf}_{wall}",
            title=f"Path Alert — {coin}USDT.P [{tf}]",
            description=wall,
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0xF97316,
        )


for tr in st.session_state.active_trades:
    _dispatch_discord_for_trade_state(tr, partner_summary=None)


def _refresh_single_item_summary(coin: str, tf: str, ctx: Dict[str, object]) -> str:
    key = f"{coin}_{tf}"
    prev_summary = st.session_state.ai_summaries.get(key)
    st.session_state.audit_logs, audit_entry = audit_and_correlate_summary(
        existing_audits=st.session_state.audit_logs,
        target_coin=coin,
        target_tf=tf,
        setup_context=ctx,
        previous_summary=prev_summary,
        active_trades=st.session_state.active_trades,
        logbook=st.session_state.logbook,
    )
    summary = run_on_demand_trader_agent(
        target_coin=coin,
        target_tf=tf,
        setup_context=ctx,
        all_committed_trades=st.session_state.active_trades,
        audit_history=st.session_state.audit_logs,
        api_key=ai_api_key,
        previous_summary=prev_summary,
        audit_entry=audit_entry,
    )
    st.session_state.ai_summaries[key] = summary
    save_summary(key, coin, tf, summary)
    return summary


def trigger_agent_for_committed_and_target(target_coin: str, target_tf: str, target_ctx: Dict[str, object]) -> None:
    """
    Master run: whenever AI is invoked from any category/setup, it updates every
    active trade AND every live scanned setup with its own individual context.
    """
    primary_summary = _refresh_single_item_summary(target_coin, target_tf, target_ctx)

    matched_tr = next(
        (t for t in st.session_state.active_trades if t.get("coin") == target_coin and t.get("timeframe") == target_tf),
        None,
    )
    if discord_enabled:
        if matched_tr:
            _dispatch_discord_for_trade_state(matched_tr, partner_summary=primary_summary)
        else:
            send_discord_webhook_alert(
                event_key=f"signal_partner_{target_coin}_{target_tf}_{target_ctx.get('entry_d')}",
                title=f"Partner Setup Read — {target_coin}USDT.P [{target_tf}]",
                description=primary_summary[:1800],
                fields=[
                    {"name": "Prob Score", "value": f"{target_ctx.get('total_score', 0)}/100", "inline": True},
                    {"name": "Primary SL", "value": fmt_price(target_ctx.get("sl") if isinstance(target_ctx.get("sl"), (int, float)) else None), "inline": True},
                    {"name": "TP1 / TP2 / TP3", "value": f"{fmt_price(target_ctx.get('tp1') if isinstance(target_ctx.get('tp1'), (int, float)) else None)} / {fmt_price(target_ctx.get('tp2') if isinstance(target_ctx.get('tp2'), (int, float)) else None)} / {fmt_price(target_ctx.get('tp3') if isinstance(target_ctx.get('tp3'), (int, float)) else None)}", "inline": False},
                ],
                webhook_url_override=discord_webhook_url,
                color=0x00C853 if target_ctx.get("pattern_type") == "BULLISH" else 0xFF3B3B,
                force=True,
            )

    visited = {(target_coin, target_tf)}

    for tr in st.session_state.active_trades:
        c = str(tr.get("coin", ""))
        tf = str(tr.get("timeframe", ""))
        if not c or not tf or (c, tf) in visited:
            continue
        visited.add((c, tf))
        tr_ctx = dict(tr)
        tr_ctx["is_committed"] = True
        _refresh_single_item_summary(c, tf, tr_ctx)

    for sc in deduped_results:
        if sc.status_label == "IDLE":
            continue
        if (sc.coin, sc.timeframe) in visited:
            continue
        visited.add((sc.coin, sc.timeframe))
        sc_ctx: Dict[str, object] = {
            "coin": sc.coin,
            "timeframe": sc.timeframe,
            "pattern_type": sc.pattern_type,
            "status_label": sc.status_label,
            "sig_code": sc.sig_code,
            "conf_score": sc.conf_score,
            "conf_pct": sc.conf_pct,
            "total_score": sc.total_score,
            "is_committed": (sc.coin, sc.timeframe) in committed_keys,
            "entry_d": sc.entry_d,
            "optimal_zone_low": sc.optimal_zone_low,
            "optimal_zone_high": sc.optimal_zone_high,
            "reversal_zone_low": sc.optimal_zone_low,
            "reversal_zone_high": sc.optimal_zone_high,
            "sl": sc.sl,
            "secondary_sl": sc.secondary_sl,
            "break_even_sl": sc.break_even_sl,
            "tp1": sc.tp1,
            "tp2": sc.tp2,
            "tp3": sc.tp3,
            "bias_change_level": sc.bias_change_level,
            "tp_obstacle_note": sc.tp_obstacle_note,
            "bad_confluences": sc.bad_confluences,
            "mtf_notes": sc.mtf_notes,
        }
        _refresh_single_item_summary(sc.coin, sc.timeframe, sc_ctx)


# 1. ACTIVE TRADE MONITORING TERMINAL
def _render_active_monitor_body() -> None:
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
                    coin = str(trade["coin"])
                    tf = str(trade["timeframe"])
                    direction = str(trade["pattern_type"])
                    live_p = last_price_by_coin.get(coin) or trade.get("last_price")
                    trade["last_price"] = live_p
                    sl_mode, active_sl_val, sl_action_note = evaluate_sl_and_tp_posture(trade)
                    trade["sl_mode"] = sl_mode
                    trade["active_sl_guidance"] = active_sl_val
                    trade["sl_action_note"] = sl_action_note

                    entry_p = trade.get("entry_d")
                    sl_p = trade.get("sl")
                    tp1_p, tp2_p, tp3_p = trade.get("tp1"), trade.get("tp2"), trade.get("tp3")
                    sl_hit = bool(trade.get("sl_hit", False))
                    all_tps_hit = bool(trade.get("tp1_hit") and trade.get("tp2_hit") and trade.get("tp3_hit"))
                    tf_h = TF_HOURS.get(tf, 1.0)
                    liq_snap = fetch_liquidation_snapshot(coin, live_p if isinstance(live_p, (int, float)) else None)
                    s_key = f"{coin}_{tf}"

                    with st.container(border=True):
                        c1, c2, c3, c4, c5 = st.columns([1.7, 2.1, 2.5, 2.0, 0.9])
                        with c1:
                            st.markdown(f"### {coin}USDT.P · `{tf}`")
                            st.markdown(f"**{direction}** · `{trade.get('sig_code', 'C')}` ({trade.get('conf_pct', 0.0):.0f}%)")
                            st.caption(f"Prob Score: **{trade.get('total_score', 0)}/100**")
                            st.caption(f"Feed: {source_used_by_coin.get(coin, 'Swap')}")
                            st.caption("COMMITTED")

                        with c2:
                            st.markdown(f"**Live:** {fmt_price(live_p if isinstance(live_p, (int, float)) else None)}")
                            st.markdown(f"**Committed Entry:** {fmt_price(entry_p if isinstance(entry_p, (int, float)) else None)}")
                            st.markdown(f"**Active SL ({sl_mode}):** `{fmt_price(active_sl_val if isinstance(active_sl_val, (int, float)) else None)}`")
                            st.caption(f"Primary SL: {fmt_price(sl_p if isinstance(sl_p, (int, float)) else None)} | Secondary SL: {fmt_price(trade.get('secondary_sl') if isinstance(trade.get('secondary_sl'), (int, float)) else None)}")
                            st.caption(f"Break-Even SL: {fmt_price(trade.get('break_even_sl') if isinstance(trade.get('break_even_sl'), (int, float)) else None)}")

                        with c3:
                            if sl_hit:
                                st.error(f"SL HIT — Logged for backtest (exits in up to {post_exit_grace_hrs:g}h)")
                            elif all_tps_hit:
                                st.success(f"TP HIT — All targets complete; logged for backtest (exits in up to {post_exit_grace_hrs:g}h)")
                            else:
                                for tp_label, tp_val, hit_flag, mult_h in (
                                    ("TP1", tp1_p, trade.get("tp1_hit", False), 4.0 * tf_h),
                                    ("TP2", tp2_p, trade.get("tp2_hit", False), 8.0 * tf_h),
                                    ("TP3", tp3_p, trade.get("tp3_hit", False), 16.0 * tf_h),
                                ):
                                    prog = 100.0 if hit_flag else compute_tp_progress(
                                        live_p if isinstance(live_p, (int, float)) else None,
                                        entry_p if isinstance(entry_p, (int, float)) else None,
                                        tp_val if isinstance(tp_val, (int, float)) else None,
                                        sl_p if isinstance(sl_p, (int, float)) else None,
                                        direction,
                                    )
                                    eta_str = "HIT" if hit_flag else format_eta_pkt(mult_h * (1.0 - prog / 100.0))
                                    if tp_label == "TP2":
                                        trade["eta_tp2"] = eta_str
                                    elif tp_label == "TP3":
                                        trade["eta_tp3"] = eta_str
                                    st.progress(int(round(prog)), text=f"{tp_label} ({fmt_price(tp_val if isinstance(tp_val, (int, float)) else None)}): {prog:.0f}% · {eta_str}")

                        with c4:
                            if sl_hit:
                                st.error("SL HIT")
                            elif all_tps_hit:
                                st.success("TP HIT")
                            else:
                                st.info(sl_action_note)
                                if trade.get("bad_confluences"):
                                    st.warning("Confluence Watch: " + ", ".join(trade["bad_confluences"]))
                                else:
                                    st.success("Confluences Supporting")

                            st.caption(f"Wall: {trade.get('tp_obstacle_note', 'Clear path')}")
                            st.caption(f"Liq: {liq_snap.get('liq_note', '-')}")

                        with c5:
                            if st.button("Kill", key=f"kill_{tab_idx}_{idx}_{coin}_{tf}", use_container_width=True):
                                st.session_state.active_trades = [
                                    t
                                    for t in st.session_state.active_trades
                                    if not (t["coin"] == coin and t["timeframe"] == tf and t["pattern_type"] == direction)
                                ]
                                save_active_trades(st.session_state.active_trades)
                                st.rerun()

                        has_sum = s_key in st.session_state.ai_summaries
                        btn_lbl = "Update All Categories (Master AI)" if has_sum else "Run All Categories (Master AI)"
                        with st.expander(f"Trading Partner Read — {coin} [{tf}]", expanded=False):
                            b_col1, b_col2 = st.columns([1.6, 1.2])
                            with b_col1:
                                if st.button(btn_lbl, key=f"run_ai_tr_{tab_idx}_{idx}_{coin}_{tf}"):
                                    ctx_tr = dict(trade)
                                    ctx_tr["is_committed"] = True
                                    trigger_agent_for_committed_and_target(coin, tf, ctx_tr)
                                    st.rerun()
                            with b_col2:
                                toggle_vis = st.checkbox(
                                    "Show Summary Text",
                                    value=st.session_state.show_summary_toggle,
                                    key=f"tog_tr_{tab_idx}_{idx}_{coin}_{tf}",
                                )
                                st.session_state.show_summary_toggle = toggle_vis

                            if st.session_state.show_summary_toggle and has_sum:
                                st.info(st.session_state.ai_summaries[s_key])
                            elif not has_sum:
                                st.caption("Click **Run All Categories** to update every trade & signal with its own read.")
    else:
        st.info("No active trades locked yet. Lock a CONFIRMED signal from the Bat Pattern tab below.")


if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=int(refresh_sec))
    def _live_monitor_fragment() -> None:
        _render_active_monitor_body()

    _live_monitor_fragment()
else:
    _render_active_monitor_body()

# 2. PATTERN TABS
st.divider()
pattern_tabs = st.tabs(["Bat Pattern", "More Patterns (Soon)"])

with pattern_tabs[0]:
    col_ready, col_conf = st.columns(2)

    ready_items = [
        r for r in deduped_results if r.status_label == "READY TO BE" and (show_overstretched or not r.overstretched)
    ]
    conf_items = [r for r in deduped_results if r.status_label == "CONFIRMED"]

    def render_signal_panel(title: str, items: List[BatScanResult], is_confirmed: bool) -> None:
        st.subheader(f"{title} ({len(items)})")
        st.caption(f"Timeframes: {format_tf_counts(items, selected_tfs)}")
        if not items:
            st.info(f"No {title.lower()} setups right now.")
            return

        for idx, r in enumerate(items):
            badge_color = "#00C853" if r.pattern_type == "BULLISH" else "#FF3B3B"
            is_committed_already = (r.coin, r.timeframe) in committed_keys
            with st.container(border=True):
                st.markdown(
                    f"<span style='color:{badge_color};font-weight:700;'>{r.coin}USDT.P · {r.pattern_type} · [{r.timeframe}]</span> "
                    f"— **`{r.sig_code}`** ({r.status_desc}) | Conf: **{r.conf_pct:.0f}%** | Prob Score: **{r.total_score}/100**",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Entry (D):** `{fmt_price(r.entry_d)}` · "
                    f"**Primary ATR SL:** `{fmt_price(r.sl)}` · "
                    f"**TP1:** `{fmt_price(r.tp1)}` · "
                    f"**TP2/3:** `{fmt_price(r.tp2)} / {fmt_price(r.tp3)}`"
                )
                if not is_committed_already and r.zone_low is not None and r.zone_high is not None:
                    st.caption(
                        f"**Prime Reversal / Entry Zone:** {fmt_price(r.optimal_zone_low)} – {fmt_price(r.optimal_zone_high)} "
                        f"({r.optimal_entry_status}) | {r.zone_label}: {fmt_price(r.zone_low)} – {fmt_price(r.zone_high)}"
                    )
                elif is_committed_already:
                    st.caption("Trade committed — optimal entry prompt hidden; following active trade plan above.")

                st.caption(
                    f"Secondary SL: {fmt_price(r.secondary_sl)} | Break-Even SL: {fmt_price(r.break_even_sl)} | "
                    f"Support: {fmt_price(r.nearest_support)} | Resistance: {fmt_price(r.nearest_resistance)} | {r.tp_obstacle_note}"
                )
                for n in r.mtf_notes:
                    st.caption(f"• {n}")

                if not is_committed_already and r.beyond_optimal_entry:
                    st.warning(f"Beyond Prime Zone: {r.optimal_entry_status}")
                if r.bad_confluences:
                    st.caption("Confluence flags: " + ", ".join(r.bad_confluences))

                if is_confirmed:
                    if is_committed_already:
                        st.success(f"Committed in Active Trades ({r.coin} [{r.timeframe}])")
                    elif st.button(f"Lock [{r.timeframe}] {r.coin} into Active Trades", key=f"lock_{r.coin}_{r.timeframe}_{r.pattern_type}_{idx}"):
                        new_trade: Dict[str, object] = {
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
                            "optimal_zone_low": r.optimal_zone_low,
                            "optimal_zone_high": r.optimal_zone_high,
                            "entry_d": r.entry_d,
                            "sl": r.sl,
                            "secondary_sl": r.secondary_sl,
                            "break_even_sl": r.break_even_sl or r.entry_d,
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
                            "last_price": last_price_by_coin.get(r.coin) or r.last_close,
                        }
                        sl_m, a_sl, s_note = evaluate_sl_and_tp_posture(new_trade)
                        new_trade["sl_mode"] = sl_m
                        new_trade["active_sl_guidance"] = a_sl
                        new_trade["sl_action_note"] = s_note

                        st.session_state.active_trades.append(new_trade)
                        save_active_trades(st.session_state.active_trades)
                        st.session_state.logbook = sync_logbook_outcomes(
                            st.session_state.logbook,
                            st.session_state.active_trades,
                            last_price_by_coin,
                        )

                        tf_h = TF_HOURS.get(r.timeframe, 1.0)
                        if discord_enabled and r.total_score >= int(discord_low_threshold):
                            send_discord_webhook_alert(
                                event_key=f"locked_{r.coin}_{r.timeframe}_{r.pattern_type}_{r.entry_d}",
                                title=f"Active Signal Locked — {r.coin}USDT.P [{r.timeframe}]",
                                description=(
                                    f"**{r.pattern_type} (`{r.sig_code}`)** · Prob Score **{r.total_score}/100** · Conf **{r.conf_pct:.0f}%**\n"
                                    f"Prime Reversal Zone: `{fmt_price(r.optimal_zone_low)}` – `{fmt_price(r.optimal_zone_high)}`"
                                ),
                                fields=[
                                    {"name": "Entry / Live", "value": f"{fmt_price(r.entry_d)} / {fmt_price(last_price_by_coin.get(r.coin) or r.last_close)}", "inline": True},
                                    {"name": "SL Plan", "value": f"Primary `{fmt_price(r.sl)}` · Sec `{fmt_price(r.secondary_sl)}` · BE `{fmt_price(r.break_even_sl)}`", "inline": True},
                                    {"name": "TP1 / TP2 / TP3", "value": f"`{fmt_price(r.tp1)}` / `{fmt_price(r.tp2)}` ({format_eta_pkt(8.0 * tf_h)}) / `{fmt_price(r.tp3)}` ({format_eta_pkt(16.0 * tf_h)})", "inline": False},
                                    {"name": "Expected Path", "value": f"{s_note} · {r.tp_obstacle_note}", "inline": False},
                                ],
                                webhook_url_override=discord_webhook_url,
                                color=0x00C853 if r.pattern_type == "BULLISH" else 0xFF3B3B,
                                force=True,
                            )
                        st.rerun()

    with col_ready:
        render_signal_panel("Ready to Be Panel", ready_items, is_confirmed=False)

    with col_conf:
        render_signal_panel("Confirmed Panel", conf_items, is_confirmed=True)

with pattern_tabs[1]:
    st.info("Additional harmonic patterns will appear in their own tabs here as we add them.")

# 3. STACKED CHARTS
st.divider()
st.subheader("Chart & Setup Inspector")

ctl1, ctl2, ctl3 = st.columns([1.4, 1.0, 1.6])
with ctl1:
    chart_coin = st.selectbox("Swap Coin", options=coins_list, index=0)
with ctl2:
    chart_tf = st.selectbox("Timeframe", options=selected_tfs, index=0)
with ctl3:
    live_p_sel = last_price_by_coin.get(chart_coin)
    liq_info = fetch_liquidation_snapshot(chart_coin, live_p_sel)
    g_diff = gate_diff_pct_by_coin.get(chart_coin)
    diff_txt = f"Gate vs Binance diff: {g_diff:.3f}%" if isinstance(g_diff, (int, float)) else str(liq_info["liq_note"])
    st.metric(
        label=f"{chart_coin}USDT.P ({source_used_by_coin.get(chart_coin, 'Swap')})",
        value=fmt_price(live_p_sel),
        delta=diff_txt,
    )

selected_scan = result_map.get((chart_coin, chart_tf))
selected_df = df_lookup.get((chart_coin, chart_tf), pd.DataFrame())
is_selected_committed = (chart_coin, chart_tf) in committed_keys

ref_price, ref_series = fetch_web_reference_prices(chart_coin, chart_tf, limit=80)
corr_info = compute_price_correlation(selected_df, ref_series)
corr_val = corr_info.get("corr")
spread_val = corr_info.get("spread_pct")

st.caption(
    f"Feed Correlation ({corr_info.get('label')}): "
    f"**{corr_val if corr_val is not None else 'N/A'}** | "
    f"Ref Swap/Price: **{fmt_price(ref_price)}** | "
    f"Basis Spread: **{f'{spread_val:+.4f}%' if spread_val is not None else 'N/A'}**"
)

st.markdown("**1. Live TradingView Chart**")
tv_sym = f"BINANCE:{to_perp_symbol(chart_coin).replace('.P', '')}.P"
render_tradingview_chart(tv_sym, chart_tf, height=430)

st.markdown("**2. Harmonic Legs, Reversal Zone, ATR SL & Support/Resistance Chart**")
if selected_scan:
    st.plotly_chart(
        render_pattern_mapping_chart(selected_df, selected_scan, liq_info, is_committed=is_selected_committed),
        use_container_width=True,
    )

detail_tabs = st.tabs(["AI Partner Read", "Data in Bullet Points"])

with detail_tabs[0]:
    if selected_scan:
        c_key = f"{selected_scan.coin}_{selected_scan.timeframe}"
        has_csum = c_key in st.session_state.ai_summaries
        matched_trade = next(
            (t for t in st.session_state.active_trades if t.get("coin") == selected_scan.coin and t.get("timeframe") == selected_scan.timeframe),
            None,
        )
        a_col1, a_col2 = st.columns([1.6, 1.2])
        with a_col1:
            if st.button(
                "Update All Categories (Master AI)" if has_csum else "Run All Categories (Master AI)",
                key="btn_chart_ai_agent",
            ):
                ctx_payload: Dict[str, object] = {
                    "coin": selected_scan.coin,
                    "timeframe": selected_scan.timeframe,
                    "pattern_type": selected_scan.pattern_type,
                    "status_label": selected_scan.status_label,
                    "sig_code": selected_scan.sig_code,
                    "conf_score": selected_scan.conf_score,
                    "conf_pct": selected_scan.conf_pct,
                    "total_score": selected_scan.total_score,
                    "is_committed": is_selected_committed,
                    "entry_d": selected_scan.entry_d,
                    "optimal_zone_low": selected_scan.optimal_zone_low,
                    "optimal_zone_high": selected_scan.optimal_zone_high,
                    "reversal_zone_low": selected_scan.optimal_zone_low,
                    "reversal_zone_high": selected_scan.optimal_zone_high,
                    "sl": selected_scan.sl,
                    "secondary_sl": selected_scan.secondary_sl,
                    "break_even_sl": selected_scan.break_even_sl,
                    "tp1": selected_scan.tp1,
                    "tp2": selected_scan.tp2,
                    "tp3": selected_scan.tp3,
                    "bias_change_level": selected_scan.bias_change_level,
                    "tp_obstacle_note": selected_scan.tp_obstacle_note,
                    "bad_confluences": selected_scan.bad_confluences,
                    "mtf_notes": selected_scan.mtf_notes,
                    "sl_mode": matched_trade.get("sl_mode") if matched_trade else "PRIMARY SL",
                    "active_sl_guidance": matched_trade.get("active_sl_guidance") if matched_trade else selected_scan.sl,
                    "tp1_hit": matched_trade.get("tp1_hit", False) if matched_trade else False,
                    "tp2_hit": matched_trade.get("tp2_hit", False) if matched_trade else False,
                    "tp3_hit": matched_trade.get("tp3_hit", False) if matched_trade else False,
                    "sl_hit": matched_trade.get("sl_hit", False) if matched_trade else False,
                    "eta_tp2": matched_trade.get("eta_tp2", "") if matched_trade else "",
                    "eta_tp3": matched_trade.get("eta_tp3", "") if matched_trade else "",
                }
                trigger_agent_for_committed_and_target(selected_scan.coin, selected_scan.timeframe, ctx_payload)
                st.rerun()
        with a_col2:
            st.session_state.show_summary_toggle = st.checkbox(
                "Show Summary Text",
                value=st.session_state.show_summary_toggle,
                key="chk_chart_sum_toggle",
            )

        if st.session_state.show_summary_toggle and has_csum:
            st.info(st.session_state.ai_summaries[c_key])
        elif not has_csum:
            st.caption("Click **Run All Categories** to update every active trade & signal with its own read.")

with detail_tabs[1]:
    if selected_scan:
        bullets = [
            f"Swap Contract & Timeframe: {selected_scan.coin}USDT.P ({source_used_by_coin.get(selected_scan.coin, 'Swap')}) on {selected_scan.timeframe} ({selected_scan.pattern_type or 'NONE'} — {selected_scan.status_desc})",
            f"Signal & Probability Score: {selected_scan.sig_code} | Confluence {selected_scan.conf_pct:.0f}% ({selected_scan.conf_score}/3) | Score {selected_scan.total_score}/100",
        ]
        if not is_selected_committed:
            bullets.append(
                f"Entry (D): {fmt_price(selected_scan.entry_d)} | Prime Zone: {fmt_price(selected_scan.optimal_zone_low)} – {fmt_price(selected_scan.optimal_zone_high)} ({selected_scan.optimal_entry_status})"
            )
        else:
            bullets.append(f"Committed Entry: {fmt_price(selected_scan.entry_d)} (Optimal entry hidden for committed trade)")
        bullets += [
            f"Stop Levels: Primary ATR SL {fmt_price(selected_scan.sl)} | Secondary SL {fmt_price(selected_scan.secondary_sl)} | Break-Even SL {fmt_price(selected_scan.break_even_sl)} | Bias-Change {fmt_price(selected_scan.bias_change_level)}",
            f"Targets: TP1 {fmt_price(selected_scan.tp1)} | TP2 {fmt_price(selected_scan.tp2)} | TP3 {fmt_price(selected_scan.tp3)}",
            f"Walls & Obstacles: Support {fmt_price(selected_scan.nearest_support)} | Resistance {fmt_price(selected_scan.nearest_resistance)} | {selected_scan.tp_obstacle_note}",
            f"Score Adjustments: {', '.join(selected_scan.score_adjustments) if selected_scan.score_adjustments else 'None'}",
        ] + [f"MTF Note: {m}" for m in selected_scan.mtf_notes]
        for b in bullets:
            st.markdown(f"* {b}")

# 4. STATUS TABLE & SIGNAL AUDIT LOGBOOK
st.divider()
st.subheader("Status Table (All Swap Contracts × Timeframes)")

status_records = []
for sym in symbols_list:
    coin = clean_coin_name(sym)
    rec: Dict[str, str] = {"Swap": f"{coin}USDT.P"}
    for tf in selected_tfs:
        match = result_map.get((coin, tf))
        if not match or match.status_label == "IDLE":
            rec[tf] = match.status_desc if match else "Hunting Waves"
        elif match.status_label == "READY TO BE":
            rec[tf] = f"Ready to Be ({match.sig_code} {match.pattern_type[:4]} · {match.total_score})"
        else:
            rec[tf] = f"Confirmed ({match.sig_code} {match.pattern_type[:4]} · {match.total_score})"
    status_records.append(rec)

st.dataframe(pd.DataFrame(status_records), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Core Signal Audit Logbook (Backtest Archive)")
log_tabs = st.tabs(HORIZONS)
for tab, horizon in zip(log_tabs, HORIZONS):
    with tab:
        scoped = filter_logbook_records(st.session_state.logbook, horizon)
        total_logged = len(scoped)
        tp_hits = sum(
            1
            for r in scoped
            if bool(r.get("tp1_hit")) or bool(r.get("tp2_hit")) or bool(r.get("tp3_hit")) or str(r.get("outcome", "")).startswith("TP")
        )
        sl_hits = sum(1 for r in scoped if bool(r.get("sl_hit")) or "SL" in str(r.get("outcome", "")).upper())
        still_tracking = max(0, total_logged - tp_hits - sl_hits)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals Logged", total_logged)
        m2.metric("TP Hits (TP1+)", tp_hits)
        m3.metric("SL Hits", sl_hits)
        m4.metric("Still Tracking", still_tracking)

        if scoped:
            for i, r in enumerate(scoped[:30], start=1):
                closed_note = f" · Closed: `{r.get('closed_pkt')}`" if r.get("closed_pkt") else ""
                st.markdown(
                    f"**#{i} | {r.get('coin')}USDT.P [{r.get('timeframe')}] {r.get('pattern_type')}** — "
                    f"Outcome: `{r.get('outcome', 'TRACKING')}` · Stage: `{r.get('status_label')}` · "
                    f"Score: `{r.get('total_score', 0)}/100` · Logged: `{r.get('timestamp_pkt')}`{closed_note}  \n"
                    f"Entry: `{fmt_price(r.get('entry_d'))}` | ATR SL: `{fmt_price(r.get('sl'))}` | "
                    f"TP1: `{fmt_price(r.get('tp1'))}` | TP2: `{fmt_price(r.get('tp2'))}` | TP3: `{fmt_price(r.get('tp3'))}`"
                )
        else:
            st.info("No logged signals for this window yet.")
