from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from market_data import now_pkt

LOGBOOK_PATH = Path("signal_logbook.json")
HORIZONS = ["1D", "4H", "1W", "1M"]


def load_logbook() -> List[Dict[str, Any]]:
    if not LOGBOOK_PATH.exists():
        return []
    try:
        data = json.loads(LOGBOOK_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_logbook(records: List[Dict[str, Any]]) -> None:
    LOGBOOK_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def record_generated_signals(records: List[Dict[str, Any]], scan_results: List[Any]) -> List[Dict[str, Any]]:
    existing_keys = {
        (
            r.get("coin"),
            r.get("timeframe"),
            r.get("pattern_type"),
            r.get("status_label"),
            round(float(r.get("entry_d") or 0.0), 6),
        )
        for r in records
    }
    added = False
    ts_str = now_pkt().strftime("%Y-%m-%d %H:%M:%S PKT")

    for res in scan_results:
        if res.status_label not in ("READY TO BE", "CONFIRMED") or res.entry_d is None:
            continue
        key = (res.coin, res.timeframe, res.pattern_type, res.status_label, round(float(res.entry_d), 6))
        if key not in existing_keys:
            existing_keys.add(key)
            records.insert(
                0,
                {
                    "timestamp_pkt": ts_str,
                    "coin": res.coin,
                    "timeframe": res.timeframe,
                    "pattern_type": res.pattern_type,
                    "status_label": res.status_label,
                    "sig_code": res.sig_code,
                    "conf_score": res.conf_score,
                    "conf_pct": res.conf_pct,
                    "entry_d": res.entry_d,
                    "sl": res.sl,
                    "tp1": res.tp1,
                    "tp2": res.tp2,
                    "tp3": res.tp3,
                    "outcome": "OPEN",
                },
            )
            added = True

    if added:
        save_logbook(records)
    return records


def filter_logbook_by_horizon(records: List[Dict[str, Any]], horizon: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "Time (PKT)",
                "Coin",
                "TF",
                "Dir",
                "Stage",
                "Sig",
                "Conf %",
                "Entry",
                "SL",
                "TP1",
                "TP2",
                "TP3",
                "Outcome",
            ]
        )
    df = pd.DataFrame(records)
    df["parsed_dt"] = pd.to_datetime(df["timestamp_pkt"].str.replace(" PKT", "", regex=False), errors="coerce")
    cutoff = now_pkt().replace(tzinfo=None) - {
        "1D": pd.Timedelta(days=1),
        "4H": pd.Timedelta(hours=4),
        "1W": pd.Timedelta(weeks=1),
        "1M": pd.Timedelta(days=30),
    }.get(horizon, pd.Timedelta(days=1))

    sub = df[df["parsed_dt"] >= cutoff].copy()
    if sub.empty:
        sub = df.head(25).copy()

    return pd.DataFrame(
        {
            "Time (PKT)": sub["timestamp_pkt"],
            "Coin": sub["coin"],
            "TF": sub["timeframe"],
            "Dir": sub["pattern_type"],
            "Stage": sub["status_label"],
            "Sig": sub["sig_code"],
            "Conf %": sub["conf_pct"].map(lambda x: f"{x:.1f}%"),
            "Entry": sub["entry_d"],
            "SL": sub["sl"],
            "TP1": sub["tp1"],
            "TP2": sub["tp2"],
            "TP3": sub["tp3"],
            "Outcome": sub["outcome"],
        }
    )
