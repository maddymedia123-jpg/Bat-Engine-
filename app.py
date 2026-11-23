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
