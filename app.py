from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
        sorted_rows = sorted(
            [r for r in rows if isinstance(r, dict)],
            key=lambda x: float(x.get("t", 0.0)),
        )
        recs = [
            {
                "open": float(r.get("o", 0.0)),
                "high": float(r.get("h", 0.0)),
                "low": float(r.get("l", 0.0)),
                "close": float(r.get("c", 0.0)),
                "volume": float(r.get("v", r.get("sum", 0.0)) or 0.0),
            }
            for r in sorted_rows
        ]
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
        options=["Binance Swap (Matches TV)", "Gate.io Swap", "Auto (Closest to Binance)"],
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
st.caption("Pine v6 BatPattern engine on swap futures (Binance & Gate.io), auto-exit into backtest logbook on TP/SL hit, and master partner AI.")

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
    elif feed_mode == "Binance Swap (Matches TV)":
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
            elif feed_mode == "Binance Swap (Matches TV)":
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
            webhook_url_override=discord
