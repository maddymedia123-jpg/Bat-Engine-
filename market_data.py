from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st

BYBIT_BASE_URL = "https://api.bybit.com"
OKX_BASE_URL = "https://www.okx.com"
KUCOIN_BASE_URL = "https://api.kucoin.com"
PKT_TZ = ZoneInfo("Asia/Karachi")

TIMEFRAME_OPTIONS: Dict[str, str] = {
    "1H": "60",
    "2H": "120",
    "4H": "240",
    "1D": "D",
}

TF_ORDER: List[str] = ["1H", "2H", "4H", "1D"]

TF_HOURS: Dict[str, float] = {
    "1H": 1.0,
    "2H": 2.0,
    "4H": 4.0,
    "1D": 24.0,
}

TV_INTERVAL_MAP: Dict[str, str] = {
    "1H": "60",
    "2H": "120",
    "4H": "240",
    "1D": "1D",
}

# Primary perpetual symbol first (BTCUSDT.P), followed by your core list
DEFAULT_PERP_SYMBOLS: List[str] = [
    "BTCUSDT.P",
    "ETHUSDT.P",
    "SOLUSDT.P",
    "BNBUSDT.P",
    "XRPUSDT.P",
    "DOGEUSDT.P",
    "ADAUSDT.P",
    "LINKUSDT.P",
    "AVAXUSDT.P",
    "DOTUSDT.P",
    "TAOUSDT.P",
    "ARBUSDT.P",
    "SUIUSDT.P",
    "APTUSDT.P",
    "NEARUSDT.P",
    "SEIUSDT.P",
    "HYPEUSDT.P",
    "FETUSDT.P",
    "ONDOUSDT.P",
    "INJUSDT.P",
]


def now_pkt() -> datetime:
    return datetime.now(PKT_TZ)


def format_eta_pkt(hours_ahead: float) -> str:
    target = now_pkt() + timedelta(hours=max(hours_ahead, 0.25))
    return f"~{hours_ahead:.1f}h ({target.strftime('%a %H:%M PKT')})"


def clean_coin_name(symbol: str) -> str:
    s = symbol.upper().split(":")[-1].replace(".P", "").replace("-SWAP", "").replace("-USDT", "")
    for suffix in ("USDT", "USDC", "USD"):
        if s.endswith(suffix) and len(s) > len(suffix):
            return s[: -len(suffix)]
    return s


def to_perp_symbol(symbol: str) -> str:
    s = symbol.upper().split(":")[-1].replace("-", "").strip()
    if s.endswith(".P"):
        return s
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s}.P"
    return f"{s}USDT.P"


def _to_bybit_linear(symbol: str) -> str:
    return to_perp_symbol(symbol).replace(".P", "")


def _to_okx_swap(symbol: str) -> str:
    base = clean_coin_name(symbol)
    return f"{base}-USDT-SWAP"


def _to_kucoin_spot(symbol: str) -> str:
    base = clean_coin_name(symbol)
    return f"{base}-USDT"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_perp_klines(symbol: str, tf_label: str, limit: int = 250) -> pd.DataFrame:
    """
    Fetches perpetual futures OHLCV candles. Primary: Bybit linear perpetuals
    (avoids Binance blocks); fallbacks: OKX perpetual swap, then KuCoin.
    """
    empty_df = pd.DataFrame(columns=["timestamp", "open", "close", "high", "low", "volume"])
    bybit_sym = _to_bybit_linear(symbol)
    bybit_iv = TIMEFRAME_OPTIONS.get(tf_label, "60")

    # 1. Bybit linear perpetual klines
    try:
        resp = requests.get(
            f"{BYBIT_BASE_URL}/v5/market/kline",
            params={"category": "linear", "symbol": bybit_sym, "interval": bybit_iv, "limit": min(limit, 200)},
            timeout=7,
        )
        resp.raise_for_status()
        raw = ((resp.json().get("result") or {}).get("list")) or []
        if raw:
            rows = [
                {
                    "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
                for item in raw
            ]
            return pd.DataFrame(rows).sort_values("timestamp").tail(limit).reset_index(drop=True)
    except Exception:
        pass

    # 2. OKX perpetual swap fallback
    okx_bar = {"1H": "1H", "2H": "2H", "4H": "4H", "1D": "1D"}.get(tf_label, "1H")
    try:
        resp = requests.get(
            f"{OKX_BASE_URL}/api/v5/market/candles",
            params={"instId": _to_okx_swap(symbol), "bar": okx_bar, "limit": str(min(limit, 200))},
            timeout=7,
        )
        resp.raise_for_status()
        raw = resp.json().get("data") or []
        if raw:
            rows = [
                {
                    "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
                for item in raw
            ]
            return pd.DataFrame(rows).sort_values("timestamp").tail(limit).reset_index(drop=True)
    except Exception:
        pass

    # 3. KuCoin fallback
    kc_type = {"1H": "1hour", "2H": "2hour", "4H": "4hour", "1D": "1day"}.get(tf_label, "1hour")
    try:
        resp = requests.get(
            f"{KUCOIN_BASE_URL}/api/v1/market/candles",
            params={"symbol": _to_kucoin_spot(symbol), "type": kc_type},
            timeout=7,
        )
        resp.raise_for_status()
        raw = resp.json().get("data") or []
        if raw:
            rows = [
                {
                    "timestamp": pd.to_datetime(int(item[0]), unit="s", utc=True),
                    "open": float(item[1]),
                    "close": float(item[2]),
                    "high": float(item[3]),
                    "low": float(item[4]),
                    "volume": float(item[5]),
                }
                for item in raw[:limit]
            ]
            return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    except Exception:
        pass

    return empty_df


@st.cache_data(ttl=20, show_spinner=False)
def fetch_perp_last_price(symbol: str) -> Optional[float]:
    bybit_sym = _to_bybit_linear(symbol)
    try:
        resp = requests.get(
            f"{BYBIT_BASE_URL}/v5/market/tickers",
            params={"category": "linear", "symbol": bybit_sym},
            timeout=6,
        )
        resp.raise_for_status()
        items = ((resp.json().get("result") or {}).get("list")) or []
        if items and items[0].get("lastPrice") is not None:
            return float(items[0]["lastPrice"])
    except Exception:
        pass

    try:
        resp = requests.get(
            f"{OKX_BASE_URL}/api/v5/market/ticker",
            params={"instId": _to_okx_swap(symbol)},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if data and data[0].get("last") is not None:
            return float(data[0]["last"])
    except Exception:
        pass

    return None


def fetch_liquidation_snapshot(coin: str, last_price: Optional[float]) -> Dict[str, Optional[float] | str]:
    if last_price is None or last_price <= 0:
        return {
            "liq_bias": "Neutral",
            "nearest_liq_level": None,
            "liq_note": "Awaiting perp price data",
        }
    upper_liq = last_price * 1.018
    lower_liq = last_price * 0.982
    return {
        "liq_bias": "Long/Short Pockets Active",
        "upper_liq": upper_liq,
        "lower_liq": lower_liq,
        "nearest_liq_level": upper_liq if last_price >= (upper_liq + lower_liq) / 2 else lower_liq,
        "liq_note": f"Lower ~{lower_liq:.6f} / Upper ~{upper_liq:.6f}",
    }


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_stoch_rsi(
    rsi: pd.Series, stoch_len: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    min_rsi = rsi.rolling(stoch_len, min_periods=1).min()
    max_rsi = rsi.rolling(stoch_len, min_periods=1).max()
    denom = (max_rsi - min_rsi).replace(0.0, np.nan)
    raw_k = ((rsi - min_rsi) / denom * 100.0).fillna(50.0)
    k = raw_k.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(smooth_d, min_periods=1).mean()
    return k, d


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean().bfill()
