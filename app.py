from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    sl_hit = bool(trade.get("sl_hit", False))

    if sl_hit:
        return "SL HIT", primary_sl if isinstance(primary_sl, (int, float)) else None, "Stop hit—stay out and wait for a fresh structure."

    if tp2_hit:
        be_val = float(be_sl) if isinstance(be_sl, (int, float)) else (float(entry_p) if isinstance(entry_p, (int, float)) else None)
        return "BREAK-EVEN SL (TP3 FOCUS)", be_val, "TP2 hit—keep SL at Break-Even and aim for TP3 or lock gains."

    if tp1_hit:
        be_val = float(be_sl) if isinstance(be_sl, (int, float)) else (float(entry_p) if isinstance(entry_p, (int, float)) else None)
        return (
            "SHIFT SL TO BREAK-EVEN -> LOOK FOR TP2",
            be_val,
            "TP1 hit—close here if desired, or shift SL to Break-Even and look for TP2 (wait for reversal zone if re-entering).",
        )

    fav_move = False
    if isinstance(live_p, (int, float)) and isinstance(entry_p, (int, float)):
        fav_move = (direction == "BULLISH" and live_p > entry_p) or (direction == "BEARISH" and live_p < entry_p)

    if fav_move and isinstance(secondary_sl, (int, float)):
        return (
            "SECONDARY SL (TRAILING)",
            float(secondary_sl),
            "Price moving in favor—switch from Primary SL to Secondary SL.",
        )

    return (
        "PRIMARY SL",
        float(primary_sl) if isinstance(primary_sl, (int, float)) else None,
        "Hold Primary SL until price moves in favor or hits TP1.",
    )


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
    st.subheader("Live Scanning (Non-Blocking)")
    auto_refresh = st.checkbox("Auto-update active monitor fragment", value=False)
    refresh_sec = st.slider("Fragment update interval (sec)", min_value=15, max_value=180, value=30, step=5)

    st.divider()
    st.subheader("Discord Bot Webhook")
    discord_webhook_url = st.text_input(
        "Discord Webhook URL (or set DISCORD_WEBHOOK_URL in Secrets)",
        type="password",
        placeholder="https://discord.com/api/webhooks/...",
    ).strip()
    discord_enabled = st.checkbox("Enable Discord Bot Alerts", value=True)

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
st.caption("Pine v6 BatPattern engine on swap futures (defaulting to BTCUSDT.P) with live RSI/Stoch, SL progression, logbook TP/SL sync, and Discord partner alerts.")

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
committed_keys = {(t.get("coin"), t.get("timeframe")) for t in st.session_state.active_trades}

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
        trade["rsi_val"] = latest_scan.rsi_val
        trade["stoch_k_val"] = latest_scan.stoch_k_val
        trade["stoch_d_val"] = latest_scan.stoch_d_val
        trade["momentum_note"] = latest_scan.momentum_note
        trade["beyond_optimal_entry"] = latest_scan.beyond_optimal_entry
        trade["optimal_entry_status"] = latest_scan.optimal_entry_status
        trade["secondary_sl"] = latest_scan.secondary_sl
        trade["break_even_sl"] = latest_scan.break_even_sl or trade.get("entry_d")
        trade["bias_change_level"] = latest_scan.bias_change_level
        trade["mtf_notes"] = latest_scan.mtf_notes
        trade["tp_obstacle_note"] = latest_scan.tp_obstacle_note

    if live_price is not None and not trade.get("sl_hit", False):
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

    sl_mode, active_sl_val, sl_action_note = evaluate_sl_and_tp_posture(trade)
    trade["sl_mode"] = sl_mode
    trade["active_sl_guidance"] = active_sl_val
    trade["sl_action_note"] = sl_action_note

if trades_changed:
    save_active_trades(st.session_state.active_trades)

st.session_state.logbook = sync_logbook_outcomes(
    st.session_state.logbook,
    st.session_state.active_trades,
    last_price_by_coin,
)


def _dispatch_discord_for_trade_state(trade: Dict[str, object], partner_summary: Optional[str] = None) -> None:
    if not discord_enabled:
        return
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
    bad_confs = trade.get("bad_confluences") or []
    wall = str(trade.get("tp_obstacle_note") or "Clear path")
    tf_h = TF_HOURS.get(tf, 1.0)
    eta_tp2 = str(trade.get("eta_tp2") or format_eta_pkt(8.0 * tf_h))
    eta_tp3 = str(trade.get("eta_tp3") or format_eta_pkt(16.0 * tf_h))

    fields = [
        {"name": "Entry / Live", "value": f"{fmt_price(entry_p if isinstance(entry_p, (int, float)) else None)} -> {fmt_price(live_p if isinstance(live_p, (int, float)) else None)}", "inline": True},
        {"name": "SL Posture", "value": f"{sl_mode} ({fmt_price(active_sl if isinstance(active_sl, (int, float)) else None)})", "inline": True},
        {"name": "Momentum", "value": str(trade.get("momentum_note") or f"RSI {trade.get('rsi_val', 50)} · Stoch {trade.get('stoch_k_val', 50)}/{trade.get('stoch_d_val', 50)}"), "inline": False},
        {"name": "Targets & ETA", "value": f"TP1 {fmt_price(trade.get('tp1'))} | TP2 {fmt_price(trade.get('tp2'))} ({eta_tp2}) | TP3 {fmt_price(trade.get('tp3'))} ({eta_tp3})", "inline": False},
        {"name": "Partner Plan", "value": str(trade.get("sl_action_note") or "-"), "inline": False},
    ]

    if partner_summary:
        send_discord_webhook_alert(
            event_key=f"partner_{coin}_{tf}_{sl_mode}_{int(tp1_hit)}{int(tp2_hit)}",
            title=f"Partner Update — {coin}USDT.P [{tf}] ({direction})",
            description=partner_summary[:1800],
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0x3B82F6,
            force=True,
        )

    if sl_hit:
        send_discord_webhook_alert(
            event_key=f"sl_hit_{coin}_{tf}",
            title=f"SL Hit — {coin}USDT.P [{tf}] ({direction})",
            description=str(trade.get("sl_action_note") or "Stop loss triggered."),
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0xEF4444,
        )
    elif tp3_hit:
        send_discord_webhook_alert(
            event_key=f"tp3_hit_{coin}_{tf}",
            title=f"TP3 Hit — {coin}USDT.P [{tf}] ({direction})",
            description="Final target TP3 reached.",
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0x10B981,
        )
    elif tp2_hit:
        send_discord_webhook_alert(
            event_key=f"tp2_hit_{coin}_{tf}",
            title=f"TP2 Hit -> Break-Even & TP3 Watch — {coin}USDT.P [{tf}]",
            description=f"TP2 hit. Hold Break-Even SL and watch TP3 ({eta_tp3}).",
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0x22C55E,
        )
    elif tp1_hit:
        send_discord_webhook_alert(
            event_key=f"tp1_hit_{coin}_{tf}",
            title=f"TP1 Hit -> Shift SL to Break-Even & Target TP2 — {coin}USDT.P [{tf}]",
            description=f"TP1 hit. Close if locking gains, or shift SL to Break-Even and look for TP2 ({eta_tp2}).",
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

    if bad_confs or (wall != "Clear path"):
        warn_reasons = list(bad_confs) + ([wall] if wall != "Clear path" else [])
        send_discord_webhook_alert(
            event_key=f"warn_{coin}_{tf}_{'|'.join(warn_reasons)}",
            title=f"Price / Confluence Warning — {coin}USDT.P [{tf}]",
            description=" | ".join(warn_reasons),
            fields=fields,
            webhook_url_override=discord_webhook_url,
            color=0xF97316,
        )


for tr in st.session_state.active_trades:
    _dispatch_discord_for_trade_state(tr, partner_summary=None)


def trigger_agent_for_committed_and_target(target_coin: str, target_tf: str, target_ctx: Dict[str, object]) -> None:
    t_key = f"{target_coin}_{target_tf}"
    prev_summary = st.session_state.ai_summaries.get(t_key)

    st.session_state.audit_logs, t_audit = audit_and_correlate_summary(
        existing_audits=st.session_state.audit_logs,
        target_coin=target_coin,
        target_tf=target_tf,
        setup_context=target_ctx,
        previous_summary=prev_summary,
        active_trades=st.session_state.active_trades,
        logbook=st.session_state.logbook,
    )

    t_summary = run_on_demand_trader_agent(
        target_coin=target_coin,
        target_tf=target_tf,
        setup_context=target_ctx,
        all_committed_trades=st.session_state.active_trades,
        audit_history=st.session_state.audit_logs,
        api_key=ai_api_key,
        previous_summary=prev_summary,
        audit_entry=t_audit,
    )
    st.session_state.ai_summaries[t_key] = t_summary
    save_summary(t_key, target_coin, target_tf, t_summary)

    matched_tr = next(
        (t for t in st.session_state.active_trades if t.get("coin") == target_coin and t.get("timeframe") == target_tf),
        None,
    )
    if matched_tr and discord_enabled:
        _dispatch_discord_for_trade_state(matched_tr, partner_summary=t_summary)
    elif discord_enabled:
        send_discord_webhook_alert(
            event_key=f"signal_partner_{t_key}_{target_ctx.get('entry_d')}",
            title=f"Partner Setup Read — {target_coin}USDT.P [{target_tf}]",
            description=t_summary[:1800],
            fields=[
                {"name": "Entry / Zone", "value": f"{fmt_price(target_ctx.get('entry_d') if isinstance(target_ctx.get('entry_d'), (int, float)) else None)} ({fmt_price(target_ctx.get('optimal_zone_low') if isinstance(target_ctx.get('optimal_zone_low'), (int, float)) else None)} - {fmt_price(target_ctx.get('optimal_zone_high') if isinstance(target_ctx.get('optimal_zone_high'), (int, float)) else None)})", "inline": True},
                {"name": "Primary SL", "value": fmt_price(target_ctx.get("sl") if isinstance(target_ctx.get("sl"), (int, float)) else None), "inline": True},
                {"name": "TP1 / TP2 / TP3", "value": f"{fmt_price(target_ctx.get('tp1') if isinstance(target_ctx.get('tp1'), (int, float)) else None)} / {fmt_price(target_ctx.get('tp2') if isinstance(target_ctx.get('tp2'), (int, float)) else None)} / {fmt_price(target_ctx.get('tp3') if isinstance(target_ctx.get('tp3'), (int, float)) else None)}", "inline": False},
            ],
            webhook_url_override=discord_webhook_url,
            color=0x00C853 if target_ctx.get("pattern_type") == "BULLISH" else 0xFF3B3B,
            force=True,
        )

    for tr in st.session_state.active_trades:
        c = str(tr.get("coin", ""))
        tf = str(tr.get("timeframe", ""))
        if not c or not tf or (c == target_coin and tf == target_tf):
            continue
        ckey = f"{c}_{tf}"
        c_prev = st.session_state.ai_summaries.get(ckey)
        st.session_state.audit_logs, c_audit = audit_and_correlate_summary(
            existing_audits=st.session_state.audit_logs,
            target_coin=c,
            target_tf=tf,
            setup_context=tr,
            previous_summary=c_prev,
            active_trades=st.session_state.active_trades,
            logbook=st.session_state.logbook,
        )
        csum = run_on_demand_trader_agent(
            target_coin=c,
            target_tf=tf,
            setup_context=tr,
            all_committed_trades=st.session_state.active_trades,
            audit_history=st.session_state.audit_logs,
            api_key=ai_api_key,
            previous_summary=c_prev,
            audit_entry=c_audit,
        )
        st.session_state.ai_summaries[ckey] = csum
        save_summary(ckey, c, tf, csum)


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
                    coin = trade["coin"]
                    tf = trade["timeframe"]
                    direction = trade["pattern_type"]
                    live_p = fetch_perp_last_price(to_perp_symbol(coin)) or trade.get("last_price")
                    trade["last_price"] = live_p
                    sl_mode, active_sl_val, sl_action_note = evaluate_sl_and_tp_posture(trade)
                    trade["sl_mode"] = sl_mode
                    trade["active_sl_guidance"] = active_sl_val
                    trade["sl_action_note"] = sl_action_note

                    entry_p = trade.get("entry_d")
                    sl_p = trade.get("sl")
                    tp1_p, tp2_p, tp3_p = trade.get("tp1"), trade.get("tp2"), trade.get("tp3")
                    sl_hit = bool(trade.get("sl_hit", False))
                    any_tp_hit = bool(trade.get("tp1_hit") or trade.get("tp2_hit") or trade.get("tp3_hit"))
                    tf_h = TF_HOURS.get(tf, 1.0)
                    liq_snap = fetch_liquidation_snapshot(coin, live_p)
                    s_key = f"{coin}_{tf}"

                    with st.container(border=True):
                        c1, c2, c3, c4, c5 = st.columns([1.7, 2.1, 2.5, 2.0, 0.9])
                        with c1:
                            st.markdown(f"### {coin}USDT.P · `{tf}`")
                            st.markdown(f"**{direction}** · `{trade.get('sig_code', 'C')}` ({trade.get('conf_pct', 0.0):.0f}%)")
                            st.caption(f"Score: **{trade.get('total_score', 0)}/100**")
                            st.caption(str(trade.get("momentum_note") or f"RSI {trade.get('rsi_val', 50)} · Stoch {trade.get('stoch_k_val', 50)}/{trade.get('stoch_d_val', 50)}"))
                            st.caption("COMMITTED")

                        with c2:
                            st.markdown(f"**Live:** {fmt_price(live_p)}")
                            st.markdown(f"**Committed Entry:** {fmt_price(entry_p)}")
                            st.markdown(f"**Active SL ({sl_mode}):** `{fmt_price(active_sl_val)}`")
                            st.caption(f"Primary SL: {fmt_price(sl_p)} | Secondary SL: {fmt_price(trade.get('secondary_sl'))}")
                            st.caption(f"Break-Even SL: {fmt_price(trade.get('break_even_sl'))}")

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
                                    if tp_label == "TP2":
                                        trade["eta_tp2"] = eta_str
                                    elif tp_label == "TP3":
                                        trade["eta_tp3"] = eta_str
                                    st.progress(int(round(prog)), text=f"{tp_label} ({fmt_price(tp_val)}): {prog:.0f}% · {eta_str}")

                        with c4:
                            st.info(sl_action_note)
                            if trade.get("bad_confluences"):
                                st.warning("Watch: " + ", ".join(trade["bad_confluences"]))
                            else:
                                st.success("Momentum & Structure Aligned")

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
                        btn_lbl = "Update Partner Read (Sync)" if has_sum else "Run Partner Read"
                        with st.expander(f"Trading Partner Read — {coin} [{tf}]", expanded=False):
                            b_col1, b_col2 = st.columns([1.4, 1.2])
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
                                st.caption("Click **Run Partner Read** for live trade guidance & Discord sync.")
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
                    f"— **`{r.sig_code}`** ({r.status_desc}) | Conf: **{r.conf_pct:.0f}%** | Score: **{r.total_score}/100**",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Entry (D):** `{fmt_price(r.entry_d)}` · "
                    f"**Primary ATR SL:** `{fmt_price(r.sl)}` · "
                    f"**TP1:** `{fmt_price(r.tp1)}` · "
                    f"**TP2/3:** `{fmt_price(r.tp2)} / {fmt_price(r.tp3)}`"
                )
                st.caption(r.momentum_note)
                if not is_committed_already and r.zone_low is not None and r.zone_high is not None:
                    st.caption(
                        f"**Optimal Entry / Reversal Zone:** {fmt_price(r.optimal_zone_low)} – {fmt_price(r.optimal_zone_high)} "
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
                    st.warning(f"Beyond Optimal Entry: {r.optimal_entry_status}")
                if r.bad_confluences:
                    st.caption("Notes: " + ", ".join(r.bad_confluences))

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
                            "rsi_val": r.rsi_val,
                            "stoch_k_val": r.stoch_k_val,
                            "stoch_d_val": r.stoch_d_val,
                            "momentum_note": r.momentum_note,
                            "beyond_optimal_entry": r.beyond_optimal_entry,
                            "optimal_entry_status": r.optimal_entry_status,
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
                        if discord_enabled:
                            send_discord_webhook_alert(
                                event_key=f"locked_{r.coin}_{r.timeframe}_{r.pattern_type}_{r.entry_d}",
                                title=f"New Active Signal Locked — {r.coin}USDT.P [{r.timeframe}]",
                                description=(
                                    f"**{r.pattern_type} (`{r.sig_code}`)** · Confidence **{r.conf_pct:.0f}%** · Score **{r.total_score}/100**\n"
                                    f"{r.momentum_note}\n"
                                    f"Reversal Zone: `{fmt_price(r.optimal_zone_low)}` – `{fmt_price(r.optimal_zone_high)}`"
                                ),
                                fields=[
                                    {"name": "Entry / Live", "value": f"{fmt_price(r.entry_d)} / {fmt_price(last_price_by_coin.get(r.coin) or r.last_close)}", "inline": True},
                                    {"name": "SL Plan", "value": f"Primary `{fmt_price(r.sl)}` · Sec `{fmt_price(r.secondary_sl)}` · BE `{fmt_price(r.break_even_sl)}`", "inline": True},
                                    {"name": "TP1 / TP2 / TP3", "value": f"`{fmt_price(r.tp1)}` / `{fmt_price(r.tp2)}` ({format_eta_pkt(8.0 * tf_h)}) / `{fmt_price(r.tp3)}` ({format_eta_pkt(16.0 * tf_h)})", "inline": False},
                                    {"name": "Behavior & Path", "value": f"{s_note} · Wall: {r.tp_obstacle_note}", "inline": False},
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
    st.metric(
        label=f"{chart_coin}USDT.P Live Swap Price",
        value=fmt_price(live_p_sel),
        delta=str(liq_info["liq_note"]),
    )

selected_scan = result_map.get((chart_coin, chart_tf))
selected_df = df_lookup.get((chart_coin, chart_tf), pd.DataFrame())
is_selected_committed = (chart_coin, chart_tf) in committed_keys

ref_price, ref_series = fetch_web_reference_prices(chart_coin, chart_tf, limit=80)
corr_info = compute_price_correlation(selected_df, ref_series)
corr_val = corr_info.get("corr")
spread_val = corr_info.get("spread_pct")

st.caption(
    f"Swap Feed Correlation ({corr_info.get('label')}): "
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
        a_col1, a_col2 = st.columns([1.5, 1.2])
        with a_col1:
            if st.button(
                "Update Partner Read (Sync)" if has_csum else "Run Partner Read",
                key="btn_chart_ai_agent",
            ):
                ctx_payload: Dict[str, object] = {
                    "coin": selected_scan.coin,
                    "timeframe": selected_scan.timeframe,
                    "pattern_type": selected_scan.pattern_type,
                    "status_label": selected_scan.status_label,
                    "sig_code": selected_scan.sig_code,
                    "conf_pct": selected_scan.conf_pct,
                    "total_score": selected_scan.total_score,
                    "rsi_val": selected_scan.rsi_val,
                    "stoch_k_val": selected_scan.stoch_k_val,
                    "stoch_d_val": selected_scan.stoch_d_val,
                    "is_committed": is_selected_committed,
                    "entry_d": selected_scan.entry_d,
                    "optimal_zone_low": selected_scan.optimal_zone_low,
                    "optimal_zone_high": selected_scan.optimal_zone_high,
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
            st.caption("Click **Run Partner Read** for a crisp trading partner view and Discord sync.")

with detail_tabs[1]:
    if selected_scan:
        bullets = [
            f"Swap Contract & Timeframe: {selected_scan.coin}USDT.P on {selected_scan.timeframe} ({selected_scan.pattern_type or 'NONE'} — {selected_scan.status_desc})",
            f"Signal & Score: {selected_scan.sig_code} | Confidence {selected_scan.conf_pct:.0f}% ({selected_scan.conf_score}/3) | Total Score {selected_scan.total_score}/100",
            f"Momentum: {selected_scan.momentum_note}",
        ]
        if not is_selected_committed:
            bullets.append(
                f"Entry (D): {fmt_price(selected_scan.entry_d)} | Optimal Zone: {fmt_price(selected_scan.optimal_zone_low)} – {fmt_price(selected_scan.optimal_zone_high)} ({selected_scan.optimal_entry_status})"
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
st.subheader("Core Signal Audit Logbook")
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
                st.markdown(
                    f"**#{i} | {r.get('coin')}USDT.P [{r.get('timeframe')}] {r.get('pattern_type')}** — "
                    f"Outcome: `{r.get('outcome', 'TRACKING')}` · Stage: `{r.get('status_label')}` · "
                    f"Score: `{r.get('total_score', 0)}/100` · Logged: `{r.get('timestamp_pkt')}`  \n"
                    f"Entry: `{fmt_price(r.get('entry_d'))}` | ATR SL: `{fmt_price(r.get('sl'))}` | "
                    f"TP1: `{fmt_price(r.get('tp1'))}` | TP2: `{fmt_price(r.get('tp2'))}` | TP3: `{fmt_price(r.get('tp3'))}`"
                )
        else:
            st.info("No logged signals for this window yet.")
