from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import requests
import streamlit as st

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"

HORIZONS: List[str] = ["24H", "3D", "7D", "All"]
PKT_TZ = ZoneInfo("Asia/Karachi")
LOCAL_STORE_DIR = Path(".local_harmonics_store")


def _get_secret(key: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None:
                return str(val).strip()
    except Exception:
        pass
    return str(os.environ.get(key, default) or default).strip()


def turso_configured() -> bool:
    url = _get_secret("TURSO_DATABASE_URL", "")
    token = _get_secret("TURSO_AUTH_TOKEN", "")
    return bool(url and token)


def _ensure_local_dir() -> None:
    LOCAL_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(name: str, default: Any) -> Any:
    _ensure_local_dir()
    p = LOCAL_STORE_DIR / f"{name}.json"
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(name: str, payload: Any) -> None:
    _ensure_local_dir()
    p = LOCAL_STORE_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _turso_http_url() -> str:
    raw = _get_secret("TURSO_DATABASE_URL", "")
    if raw.startswith("libsql://"):
        raw = "https://" + raw[len("libsql://") :]
    return raw.rstrip("/")


def _exec_turso(statements: List[Dict[str, Any]]) -> bool:
    if not turso_configured():
        return False
    url = f"{_turso_http_url()}/v2/pipeline"
    token = _get_secret("TURSO_AUTH_TOKEN", "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    requests_body = [
        {"type": "execute", "stmt": {"sql": s["sql"], "args": s.get("args", [])}}
        for s in statements
    ] + [{"type": "close"}]
    try:
        resp = requests.post(url, headers=headers, json={"requests": requests_body}, timeout=10)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def init_storage() -> None:
    _ensure_local_dir()
    _exec_turso(
        [
            {"sql": "CREATE TABLE IF NOT EXISTS active_trades (id TEXT PRIMARY KEY, payload TEXT)"},
            {"sql": "CREATE TABLE IF NOT EXISTS signal_logbook (id TEXT PRIMARY KEY, ts_epoch REAL, payload TEXT)"},
            {"sql": "CREATE TABLE IF NOT EXISTS ai_summaries (cache_key TEXT PRIMARY KEY, coin TEXT, timeframe TEXT, summary TEXT)"},
            {"sql": "CREATE TABLE IF NOT EXISTS audit_logs (id TEXT PRIMARY KEY, ts_epoch REAL, payload TEXT)"},
        ]
    )


def load_active_trades() -> List[Dict[str, Any]]:
    data = _read_json("active_trades", [])
    return data if isinstance(data, list) else []


def save_active_trades(trades: List[Dict[str, Any]]) -> None:
    _write_json("active_trades", trades[:10])
    _exec_turso(
        [
            {"sql": "DELETE FROM active_trades"},
            *[
                {
                    "sql": "INSERT INTO active_trades (id, payload) VALUES (?, ?)",
                    "args": [
                        {"type": "text", "value": f"{t.get('coin','')}_{t.get('timeframe','')}_{t.get('pattern_type','')}"},
                        {"type": "text", "value": json.dumps(t, default=str)},
                    ],
                }
                for t in trades[:10]
            ],
        ]
    )


def load_logbook() -> List[Dict[str, Any]]:
    data = _read_json("signal_logbook", [])
    return data if isinstance(data, list) else []


def record_generated_signals(existing_logbook: List[Dict[str, Any]], results: List[Any]) -> List[Dict[str, Any]]:
    records = list(existing_logbook) if isinstance(existing_logbook, list) else []
    seen_keys = {
        f"{r.get('coin')}_{r.get('timeframe')}_{r.get('pattern_type')}_{r.get('status_label')}_{r.get('entry_d')}"
        for r in records
    }
    now_dt = datetime.now(PKT_TZ)
    now_ts = now_dt.timestamp()
    pkt_str = now_dt.strftime("%Y-%m-%d %H:%M PKT")
    added = False

    for res in results:
        status = getattr(res, "status_label", "IDLE")
        if status not in ("READY TO BE", "CONFIRMED"):
            continue
        coin = getattr(res, "coin", "")
        tf = getattr(res, "timeframe", "")
        ptype = getattr(res, "pattern_type", "")
        entry_d = getattr(res, "entry_d", None)
        dedup_key = f"{coin}_{tf}_{ptype}_{status}_{entry_d}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        records.insert(
            0,
            {
                "coin": coin,
                "timeframe": tf,
                "pattern_type": ptype,
                "status_label": status,
                "total_score": getattr(res, "total_score", 0),
                "entry_d": entry_d,
                "sl": getattr(res, "sl", None),
                "tp1": getattr(res, "tp1", None),
                "tp2": getattr(res, "tp2", None),
                "tp3": getattr(res, "tp3", None),
                "outcome": "TRACKING",
                "ts_epoch": now_ts,
                "timestamp_pkt": pkt_str,
            },
        )
        added = True

    if added:
        records = records[:300]
        _write_json("signal_logbook", records)
    return records


def filter_logbook_records(records: List[Dict[str, Any]], horizon: str) -> List[Dict[str, Any]]:
    if not records:
        return []
    if horizon == "All":
        return records
    hours_map = {"24H": 24.0, "3D": 72.0, "7D": 168.0}
    max_h = hours_map.get(horizon)
    if max_h is None:
        return records
    cutoff = datetime.now(PKT_TZ).timestamp() - (max_h * 3600.0)
    return [r for r in records if float(r.get("ts_epoch", 0.0)) >= cutoff]


def load_saved_summaries() -> Dict[str, str]:
    data = _read_json("ai_summaries", {})
    return data if isinstance(data, dict) else {}


def save_summary(cache_key: str, coin: str, timeframe: str, summary: str) -> None:
    store = load_saved_summaries()
    store[cache_key] = summary
    _write_json("ai_summaries", store)
    _exec_turso(
        [
            {
                "sql": "INSERT OR REPLACE INTO ai_summaries (cache_key, coin, timeframe, summary) VALUES (?, ?, ?, ?)",
                "args": [
                    {"type": "text", "value": cache_key},
                    {"type": "text", "value": coin},
                    {"type": "text", "value": timeframe},
                    {"type": "text", "value": summary},
                ],
            }
        ]
    )


def load_audit_logs() -> List[Dict[str, Any]]:
    data = _read_json("audit_logs", [])
    return data if isinstance(data, list) else []


def silent_self_audit_and_update(
    existing_audits: List[Dict[str, Any]],
    active_trades: List[Dict[str, Any]],
    logbook: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    audits = list(existing_audits) if isinstance(existing_audits, list) else []
    now_dt = datetime.now(PKT_TZ)
    entry = {
        "audited_pkt": now_dt.strftime("%Y-%m-%d %H:%M PKT"),
        "coin": active_trades[0].get("coin", "SCAN") if active_trades else "SCAN",
        "timeframe": active_trades[0].get("timeframe", "1H") if active_trades else "1H",
        "outcome_seen": f"{len(active_trades)} active / {len(logbook)} logged",
        "lesson_learned": "Verified coin/timeframe binding and ATR stop/target alignment.",
        "rule_adjustment": "Keep confirmation strictly on active pattern state and check HTF walls.",
    }
    audits.insert(0, entry)
    audits = audits[:50]
    _write_json("audit_logs", audits)
    return audits


def _resolve_openrouter_key(api_key: Optional[str] = None) -> str:
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    for secret_name in ("OPENROUTER_API_KEY", "AI_API_KEY", "OPENAI_API_KEY"):
        found = _get_secret(secret_name, "")
        if found:
            return found
    return ""


def _build_fallback_layman_summary(
    target_coin: str,
    target_tf: str,
    setup_context: Dict[str, Any],
    all_committed_trades: List[Dict[str, Any]],
    audit_history: List[Dict[str, Any]],
    api_note: str = "",
) -> str:
    direction = str(setup_context.get("pattern_type") or "SETUP")
    sig_code = str(setup_context.get("sig_code") or setup_context.get("status_label") or "")
    score = setup_context.get("total_score", 0)
    conf = setup_context.get("conf_pct", 0.0)
    mtf = setup_context.get("mtf_notes") or ["No conflicting higher-timeframe forming pattern"]
    wall = setup_context.get("tp_obstacle_note") or "Clear path"

    latest_audit = audit_history[0] if audit_history else None
    memory_line = (
        f"{latest_audit.get('coin')} ({latest_audit.get('timeframe')}): "
        f"{latest_audit.get('lesson_learned')} -> {latest_audit.get('rule_adjustment')}"
        if latest_audit
        else f"Monitoring {len(all_committed_trades)} committed swap trade(s) with no prior audit warnings."
    )

    header_note = f"*(OpenRouter note: {api_note} — showing structured institutional layman read)*\n\n" if api_note else ""
    return (
        f"{header_note}"
        f"**Institutional Layman Read — {target_coin} [{target_tf}] ({direction} · {sig_code}):**\n\n"
        f"* **What's Happening Now:** Score is **{score}/100** with confluence at **{conf}%**. {', '.join(str(x) for x in mtf)}.\n"
        f"* **Risk & SL Plan:** Primary ATR SL is the main invalidation line; a secondary SL can trail behind structure/TP1 once price moves in your favor.\n"
        f"* **What Could Happen Next:** Watch reaction around the D/optimal zone and obstacle status (**{wall}**) before expecting progression toward TP1–TP3.\n"
        f"* **Agent Memory Note:** {memory_line}"
    )


def run_on_demand_trader_agent(
    target_coin: str,
    target_tf: str,
    setup_context: Dict[str, Any],
    all_committed_trades: List[Dict[str, Any]],
    audit_history: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> str:
    resolved_key = _resolve_openrouter_key(api_key)
    if not resolved_key:
        return _build_fallback_layman_summary(
            target_coin,
            target_tf,
            setup_context,
            all_committed_trades,
            audit_history,
            api_note="No OpenRouter API key found in input or Streamlit Secrets (`OPENROUTER_API_KEY`)",
        )

    model = _get_secret("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL) or DEFAULT_OPENROUTER_MODEL
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _get_secret("OPENROUTER_SITE_URL", "https://harmonicsengine.streamlit.app"),
        "X-Title": _get_secret("OPENROUTER_APP_NAME", "Harmonics Engine"),
    }

    system_prompt = (
        "You are a concise institutional crypto swap futures trading assistant. "
        "Explain the harmonic Bat setup in plain, practical layman terms with bullet points covering: "
        "What's Happening Now, Risk & SL Plan, What Could Happen Next, and Portfolio/Agent Context."
    )
    user_payload = {
        "target_coin": target_coin,
        "target_timeframe": target_tf,
        "setup_context": setup_context,
        "committed_trades_count": len(all_committed_trades),
        "committed_trades": all_committed_trades[:10],
        "recent_audits": audit_history[:5],
    }

    try:
        resp = requests.post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(user_payload)},
                ],
                "temperature": 0.3,
                "max_tokens": 500,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
        if content:
            return content
        return _build_fallback_layman_summary(
            target_coin,
            target_tf,
            setup_context,
            all_committed_trades,
            audit_history,
            api_note="OpenRouter returned an empty response",
        )
    except Exception as exc:
        return _build_fallback_layman_summary(
            target_coin,
            target_tf,
            setup_context,
            all_committed_trades,
            audit_history,
            api_note=str(exc),
        )
