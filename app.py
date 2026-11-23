from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from bat_engine import BatScanResult, run_bat_engine
from market_data import (
    DEFAULT_SYMBOLS,
    TIMEFRAME_OPTIONS,
    TV_INTERVAL_MAP,
    clean_coin_name,
    compute_price_correlation,
    fetch_kucoin_klines,
    fetch_kucoin_last_price,
    fetch_liquidation_snapshot,
    fetch_web_reference_prices,
    to_kucoin_symbol,
)

st.set_page_config(page_title="Bat Harmonic Scanner (KuCoin)", layout="wide")


def fmt_price(val: Optional[float], scale: float = 1.0) -> str:
    if val is None:
        return "-"
    adj = val * scale
    return f"{adj:.6f}".rstrip("0").rstrip(".")


def render_tradingview_chart(tv_symbol: str, tf_label: str, height: int = 440) -> None:
    interval = TV_INTERVAL_MAP.get(tf_label, "60")
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_bat_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%",
          "height": {height},
          "symbol": "{tv_symbol}",
          "interval": "{interval}",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#1a2027",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "container_id": "tv_bat_chart"
        }});
      </script>
    </div>
    """
    components.html(html, height=height + 16)


with st.sidebar:
    st.header("Watchlist & Timeframes")
    raw_symbols = st.text_area(
        "Watchlist (10 coins — edit anytime)",
        value="\n".join(DEFAULT_SYMBOLS),
        height=220,
    )
    selected_tfs = st.multiselect(
        "Timeframes (Initial: 1H, 2H, 4H, 1D)",
        options=list(TIMEFRAME_OPTIONS.keys()),
        default=["1H", "2H", "4H", "1D"],
    )

    st.divider()
    st.subheader("Price Correlation Mode")
    price_mode = st.radio(
        "Display levels in:",
        options=["KuCoin Native", "Web / Reference Adjusted"],
        index=0,
    )
    show_overstretched = st.checkbox("Show overstretched READY (W) rows", value=False)

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

    if st.button("Refresh Data Now"):
        st.cache_data.clear()
        st.rerun()

symbols_list = [line.strip() for line in raw_symbols.splitlines() if line.strip()][:10]
if not symbols_list:
    symbols_list = DEFAULT_SYMBOLS
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

st.title("Bat Harmonic Scanner — KuCoin + Live Chart")
st.caption("Tracks IN MAKING (forming/ready) vs. MADE (confirmed) Bat setups, reversal zones, and KuCoin-to-Web price correlation.")

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
        results.append(run_bat_engine(df, sym, coin, tf, params))

# Top chart selector & live TradingView chart
chart_col, corr_col = st.columns([2.2, 1.0])
with chart_col:
    st.subheader("Live Chart")
    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        chart_coin = st.selectbox("Select Coin", options=list(coins_tuple), index=0)
    with c_sel2:
        chart_tf = st.selectbox("Select Timeframe", options=selected_tfs, index=0)
    tv_sym = f"KUCOIN:{to_kucoin_symbol(chart_coin).replace('-', '')}"
    render_tradingview_chart(tv_sym, chart_tf, height=410)

with corr_col:
    st.subheader("KuCoin ↔ Web Price Correlation")
    corr_rows = []
    for coin in coins_tuple:
        info = corr_by_coin[coin]
        liq = fetch_liquidation_snapshot(coin)
        corr_rows.append(
            {
                "Coin": coin,
                "KuCoin": fmt_price(info["kucoin"]),
                "Web Ref": fmt_price(info["web"]),
                "Diff %": f"{info['diff_pct']:+.2f}%" if info["diff_pct"] is not None else "-",
                "Liq Feed": liq["liq_bias"],
            }
        )
    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True, height=410)


def build_signal_rows(items: List[BatScanResult]) -> pd.DataFrame:
    rows = []
    for r in items:
        scale = 1.0
        if price_mode == "Web / Reference Adjusted" and corr_by_coin[r.coin]["ratio"]:
            scale = float(corr_by_coin[r.coin]["ratio"])

        tp1_txt = fmt_price(r.tp1, scale)
        if r.tp1_hit and r.tp1 is not None:
            tp1_txt = f"~~{tp1_txt}~~"

        tp2_txt = fmt_price(r.tp2, scale)
        if r.tp2_hit and r.tp2 is not None:
            tp2_txt = f"~~{tp2_txt}~~"
        tp3_txt = fmt_price(r.tp3, scale)
        tp23_txt = "-" if (r.tp2 is None or r.tp3 is None) else f"{tp2_txt} / {tp3_txt}"

        zone_txt = (
            f"{fmt_price(r.zone_low, scale)} – {fmt_price(r.zone_high, scale)}"
            if (r.zone_low is not None and r.zone_high is not None)
            else "-"
        )

        rows.append(
            {
                "Coin": f"{r.coin} {r.timeframe}",
                "Dir": r.pattern_type,
                "Sig": r.sig_code,
                "Entry (D)": fmt_price(r.entry_d, scale),
                "SL": fmt_price(r.sl, scale),
                "TP1": tp1_txt,
                "TP2 / TP3": tp23_txt,
                "Reversal Zone": zone_txt,
                "Zone Type": r.zone_label,
            }
        )
    return pd.DataFrame(rows)


in_making = [
    r for r in results if r.phase == "IN MAKING" and (show_overstretched or not r.overstretched)
]
made = [r for r in results if r.phase == "MADE"]

st.divider()
col_making, col_made = st.columns(2)

with col_making:
    st.subheader("In Making (Forming / READY)")
    if in_making:
        st.dataframe(build_signal_rows(in_making), use_container_width=True, hide_index=True)
    else:
        st.info("No IN MAKING patterns right now.")

with col_made:
    st.subheader("Made (CONFIRMED)")
    if made:
        st.dataframe(build_signal_rows(made), use_container_width=True, hide_index=True)
    else:
        st.info("No MADE patterns right now.")

st.divider()
st.subheader("Status Table (All Coins × Timeframes)")

status_records = []
for sym in symbols_list:
    coin = clean_coin_name(sym)
    rec: Dict[str, str] = {"Coin": coin}
    for tf in selected_tfs:
        match = next((x for x in results if x.coin == coin and x.timeframe == tf), None)
        if not match or match.phase == "IDLE":
            rec[tf] = "Idle (-)"
        elif match.phase == "IN MAKING":
            rec[tf] = f"In Making ({match.sig_code} {match.pattern_type[:4]})"
        else:
            rec[tf] = f"Made ({match.sig_code} {match.pattern_type[:4]})"
    status_records.append(rec)

st.dataframe(pd.DataFrame(status_records), use_container_width=True, hide_index=True)
