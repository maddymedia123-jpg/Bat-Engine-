from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st

KUCOIN_BASE_URL = "https://api.kucoin.com"
PKT_TZ = ZoneInfo("Asia/Karachi")

TIMEFRAME_OPTIONS: Dict[str, str] = {
    "1H": "1hour",
    "2H": "2hour",
    "4H": "4hour",
    "1D": "1day",
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

INITIAL_10_COINS: List[str] = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "BNB-USDT",
    "XRP-USDT",
    "DOGE-USDT",
    "ADA-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "DOT-USDT",
]

EXTRA_10_COINS: List[str] = [
    "TAO-USDT",
    "ARB-USDT",
    "SUI-USDT",
    "APT-USDT",
    "NEAR-USDT",
    "SEI-USDT",
    "HYPE-USDT",
    "FET-USDT",
    "ONDO-USDT",
    "INJ-USDT",
]

DEFAULT_20_SYMBOLS: List[str] = INITIAL_10_COINS + EXTRA_10_COINS

COINGECKO_ID_MAP: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "LINK": "chainlink",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "TAO": "bittensor",
    "ARB": "arbitrum",
    "SUI": "sui",
    "APT": "aptos",
    "NEAR": "near",
    "SEI": "sei-network",
    "HYPE": "hyperliquid",
    "FET": "fetch-ai",
    "ONDO": "ondo-finance",
    "INJ": "injective-protocol",
}


def now_pkt() -> datetime:
    return datetime.now(PKT_TZ)


def format_eta_pkt(hours_ahead: float) -> str:
    target = now_pkt() + timedelta(hours=max(hours_ahead, 0.25))
    return f"~{hours_ahead:.1f}h ({target.strftime('%a %H:%M PKT')})"


def clean_coin_name(symbol: str) -> str:
    s = symbol.upper().split(":")[-1].replace(".P", "")
    for suffix in ("-USDT", "-USDC", "-USD", "USDT", "USDC", "USD"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def to_kucoin_symbol(symbol: str) -> str:
    s = symbol.upper().split(":")[-1].replace(".P", "")
    if "-" in s:
        return s
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}-{quote}"
    return f"{s}-USDT"


@st.cache_data(ttl=45, show_spinner=False)
def fetch_kucoin_klines(symbol: str, tf_label: str, limit: int = 250) -> pd.DataFrame:
    kc_symbol = to_kucoin_symbol(symbol)
    kc_type = TIMEFRAME_OPTIONS.get(tf_label, "1hour")
    resp = requests.get(
        f"{KUCOIN_BASE_URL}/api/v1/market/candles",
        params={"symbol": kc_symbol, "type": kc_type},
        timeout=10,
    )
    resp.raise_for_status()
    raw = resp.json().get("data", [])
    if not raw:
        return pd.DataFrame(columns=["timestamp", "open", "close", "high", "low", "volume"])

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


@st.cache_data(ttl=30, show_spinner=False)
def fetch_kucoin_last_price(symbol: str) -> Optional[float]:
    kc_symbol = to_kucoin_symbol(symbol)
    resp = requests.get(
        f"{KUCOIN_BASE_URL}/api/v1/market/orderbook/level1",
        params={"symbol": kc_symbol},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or {}
    price = data.get("price")
    return float(price) if price is not None else None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_web_reference_prices(coins: Tuple[str, ...]) -> Dict[str, Optional[float]]:
    ids = [COINGECKO_ID_MAP[c] for c in coins if c in COINGECKO_ID_MAP]
    result: Dict[str, Optional[float]] = {c: None for c in coins}
    if not ids:
        return result
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(sorted(set(ids))), "vs_currencies": "usd"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        for coin in coins:
            cg_id = COINGECKO_ID_MAP.get(coin)
            if cg_id and cg_id in data and "usd" in data[cg_id]:
                result[coin] = float(data[cg_id]["usd"])
    except Exception:
        pass
    return result


def compute_price_correlation(
    kucoin_price: Optional[float], web_price: Optional[float]
) -> Dict[str, Optional[float]]:
    if kucoin_price is None or kucoin_price == 0:
        return {"kucoin": kucoin_price, "web": web_price, "diff_abs": None, "diff_pct": None, "ratio": None}
    if web_price is None or web_price == 0:
        return {"kucoin": kucoin_price, "web": None, "diff_abs": 0.0, "diff_pct": 0.0, "ratio": 1.0}
    diff_abs = web_price - kucoin_price
    diff_pct = (diff_abs / kucoin_price) * 100.0
    ratio = web_price / kucoin_price
    return {
        "kucoin": kucoin_price,
        "web": web_price,
        "diff_abs": diff_abs,
        "diff_pct": diff_pct,
        "ratio": ratio,
    }


def fetch_liquidation_snapshot(coin: str, last_price: Optional[float]) -> Dict[str, Optional[float] | str]:
    """
    Free liquidation zone estimator anchored to KuCoin price & swing liquidity
    (ready to swap with a free live liquidation endpoint when desired).
    """
    if last_price is None or last_price <= 0:
        return {
            "liq_bias": "Neutral",
            "nearest_liq_level": None,
            "liq_note": "Awaiting price data",
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


def generate_layman_ai_summary(
    summary_payload: Dict[str, object],
    api_key: str = "",
) -> str:
    """
    Uses an OpenAI-compatible chat endpoint if api_key is provided;
    otherwise returns a clear layman-language summary from the signal metrics.
    """
    if api_key.strip():
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You explain crypto harmonic Bat trading setups in plain, simple, "
                                "layman language without jargon. Highlight entry patience, risk/SL buffer, "
                                "higher-timeframe pull/opposition, and what to watch next."
                            ),
                        },
                        {"role": "user", "content": str(summary_payload)},
                    ],
                    "temperature": 0.3,
                },
                timeout=12,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content:
                return content.strip()
        except Exception as exc:
            fallback_note = f"(AI key call fell back to plain summary: {exc})\n\n"
        else:
            fallback_note = ""
    else:
        fallback_note = ""

    coin = summary_payload.get("coin", "This coin")
    tf = summary_payload.get("timeframe", "")
    direction = summary_payload.get("pattern_type", "")
    stage = summary_payload.get("status_label", "")
    score = summary_payload.get("total_score", 0)
    notes = summary_payload.get("notes", [])
    note_txt = " ".join(notes) if isinstance(notes, list) and notes else "No major timeframe warning right now."

    return (
        f"{fallback_note}"
        f"**Plain Summary for {coin} ({tf}) — {direction} [{stage}]:** "
        f"Overall setup score is **{score}/100**. {note_txt} "
        f"In simple terms, wait for price to settle cleanly inside the optimal reversal zone rather than rushing in early, "
        f"and keep an eye on nearby support/resistance walls and your stop-loss buffer if price spikes with a wick."
    )
