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
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]
    except Exception:
        return []


def save_logbook(records: List[Dict[str, Any]]) -> None:
    clean_records = [item for item in records if isinstance(item, dict)]
    LOGBOOK_PATH.write_text(json.dumps(clean_records, indent=2), encoding="utf-8")


def _normalize_status(res: Any) -> str:
    status = getattr(res, "status_label", None)
    if status:
        return str(status)
    phase = str(getattr(res, "phase", "")).upper()
    if phase in ("IN MAKING", "READY TO BE"):
        return "READY TO BE"
    if phase in ("MADE", "CONFIRMED"):
        return "CONFIRMED"
    return "IDLE"


def record_generated_signals(records: List[Any], scan_results: List[Any]) -> List[Dict[str, Any]]:
    valid_records: List[Dict[str, Any]] = [r for r in records if isinstance(r, dict)]
    existing_keys = {
        (
            r.get("coin"),
            r.get("timeframe"),
            r.get("pattern_type"),
            r.get("status_label"),
            round(float(r.get("entry_d") or 0.0), 6),
        )
        for r in valid_records
    }
    added = False
    ts_str = now_pkt().strftime("%Y-%m-%d %H:%M:%S PKT")

    for res in scan_results:
        status_label = _normalize_status(res)
        entry_d = getattr(res, "entry_d", None)
        if status_label not in ("READY TO BE", "CONFIRMED") or entry_d is None:
            continue

        coin = getattr(res, "coin", "")
        timeframe = getattr(res, "timeframe", "")
        pattern_type = getattr(res, "pattern_type", "")
        key = (coin, timeframe, pattern_type, status_label, round(float(entry_d), 6))

        if key not in existing_keys:
            existing_keys.add(key)
            valid_records.insert(
                0,
                {
                    "timestamp_pkt": ts_str,
                    "coin": coin,
                    "timeframe": timeframe,
                    "pattern_type": pattern_type,
                    "status_label": status_label,
                    "sig_code": getattr(res, "sig_code", "-"),
                    "conf_score": getattr(res, "conf_score", 0),
                    "conf_pct": getattr(res, "conf_pct", 0.0),
                    "entry_d": entry_d,
                    "sl": getattr(res, "sl", None),
                    "tp1": getattr(res, "tp1", None),
                    "tp2": getattr(res, "tp2", None),
                    "tp3": getattr(res, "tp3", None),
                    "outcome": "OPEN",
                },
            )
            added = True

    if added:
        save_logbook(valid_records)
    return valid_records


def filter_logbook_by_horizon(records: List[Any], horizon: str) -> pd.DataFrame:
    cols = [
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
    valid_records = [r for r in records if isinstance(r, dict)]
    if not valid_records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(valid_records)
    df["parsed_dt"] = pd.to_datetime(
        df.get("timestamp_pkt", pd.Series(dtype=str))
        .astype(str)
        .str.replace(" PKT", "", regex=False),
        errors="coerce",
    )
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
            "Time (PKT)": sub.get("timestamp_pkt", ""),
            "Coin": sub.get("coin", ""),
            "TF": sub.get("timeframe", ""),
            "Dir": sub.get("pattern_type", ""),
            "Stage": sub.get("status_label", ""),
            "Sig": sub.get("sig_code", "-"),
            "Conf %": sub.get("conf_pct", 0.0).map(lambda x: f"{float(x or 0.0):.1f}%"),
            "Entry": sub.get("entry_d"),
            "SL": sub.get("sl"),
            "TP1": sub.get("tp1"),
            "TP2": sub.get("tp2"),
            "TP3": sub.get("tp3"),
            "Outcome": sub.get("outcome", "OPEN"),
        }
    )
