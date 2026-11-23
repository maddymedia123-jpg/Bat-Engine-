from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from market_data import now_pkt

LOGBOOK_PATH = Path("signal_logbook.json")
HORIZONS = ["All Time", "1D", "1W", "1M"]


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
    ts_str = now_pkt().strftime("%a %H:%M PKT")
    iso_str = now_pkt().replace(tzinfo=None).isoformat()

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
                    "created_iso": iso_str,
                    "timestamp_pkt": ts_str,
                    "coin": coin,
                    "timeframe": timeframe,
                    "pattern_type": pattern_type,
                    "status_label": status_label,
                    "sig_code": getattr(res, "sig_code", "-"),
                    "conf_score": getattr(res, "conf_score", 0),
                    "conf_pct": getattr(res, "conf_pct", 0.0),
                    "total_score": getattr(res, "total_score", 0),
                    "entry_d": entry_d,
                    "sl": getattr(res, "sl", None),
                    "tp1": getattr(res, "tp1", None),
                    "tp2": getattr(res, "tp2", None),
                    "tp3": getattr(res, "tp3", None),
                    "outcome": "TRACKING",
                },
            )
            added = True

    if added:
        save_logbook(valid_records)
    return valid_records


def filter_logbook_records(records: List[Any], horizon: str) -> List[Dict[str, Any]]:
    valid_records = [r for r in records if isinstance(r, dict)]
    if not valid_records or horizon == "All Time":
        return valid_records

    cutoff = now_pkt().replace(tzinfo=None) - {
        "1D": pd.Timedelta(days=1),
        "1W": pd.Timedelta(weeks=1),
        "1M": pd.Timedelta(days=30),
    }.get(horizon, pd.Timedelta(days=1))

    filtered: List[Dict[str, Any]] = []
    for r in valid_records:
        dt = pd.to_datetime(r.get("created_iso"), errors="coerce")
        if pd.notna(dt) and dt >= cutoff:
            filtered.append(r)
    return filtered if filtered else valid_records[:25]
