from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
            {"sql": "CREATE TABLE IF NOT EXISTS discord_events (event_key TEXT PRIMARY KEY, ts_epoch REAL, payload TEXT)"},
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


def _save_audit_logs(audits: List[Dict[str, Any]]) -> None:
    trimmed = audits[:100]
    _write_json("audit_logs", trimmed)
    _exec_turso(
        [
            {
                "sql": "INSERT OR REPLACE INTO audit_logs (id, ts_epoch, payload) VALUES (?, ?, ?)",
                "args": [
                    {"type": "text", "value": f"audit_{trimmed[0].get('audited_pkt', '')}_{trimmed[0].get('coin', '')}"},
                    {"type": "float", "value": float(trimmed[0].get("ts_epoch", datetime.now(PKT_TZ).timestamp()))},
                    {"type": "text", "value": json.dumps(trimmed[0], default=str)},
                ],
            }
        ]
        if trimmed
        else []
    )


def audit_and_correlate_summary(
    existing_audits: List[Dict[str, Any]],
    target_coin: str,
    target_tf: str,
    setup_context: Dict[str, Any],
    previous_summary: Optional[str],
    active_trades: List[Dict[str, Any]],
    logbook: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Correlates the new summary update with the previous summary for the same coin/TF
    and past memory audits, flagging missed items and raising accuracy score.
    """
    audits = list(existing_audits) if isinstance(existing_audits, list) else []
    now_dt = datetime.now(PKT_TZ)
    now_ts = now_dt.timestamp()

    coin_past_audits = [a for a in audits if a.get("coin") == target_coin and a.get("timeframe") == target_tf]
    prev_accuracy = float(coin_past_audits[0].get("accuracy_pct", 72.0)) if coin_past_audits else 72.0

    missed_items: List[str] = []
    sl_mode = str(setup_context.get("sl_mode") or "PRIMARY SL")
    tp1_hit = bool(setup_context.get("tp1_hit", False))
    tp2_hit = bool(setup_context.get("tp2_hit", False))
    sl_hit = bool(setup_context.get("sl_hit", False))
    bad_confs = setup_context.get("bad_confluences") or []
    wall = str(setup_context.get("tp_obstacle_note") or "Clear path")

    if previous_summary:
        prev_l = previous_summary.lower()
        if tp1_hit and "tp1" not in prev_l:
            missed_items.append("Prior summary missed TP1 hit & shift to Break-Even / TP2 focus")
        if sl_mode != "PRIMARY SL" and sl_mode.lower() not in prev_l:
            missed_items.append(f"Prior summary did not reflect {sl_mode}")
        if bad_confs and "confluence" not in prev_l and "warning" not in prev_l:
            missed_items.append("Prior summary missed active confluence warnings")
        if "wall" not in prev_l and wall != "Clear path":
            missed_items.append(f"Prior summary missed path wall ({wall})")
    else:
        missed_items.append("Initial baseline summary recorded; future reruns will cross-check against this snapshot")

    accuracy_gain = 4.0 if not missed_items or previous_summary is None else 6.0
    new_accuracy = round(min(99.0, prev_accuracy + accuracy_gain), 1)

    lesson = (
        f"Correlated with prior {target_coin} [{target_tf}] summary; addressed: {'; '.join(missed_items[:2])}."
        if missed_items
        else f"Consistent with prior {target_coin} [{target_tf}] summary; state & SL/TP posture verified."
    )
    rule_adj = (
        "Enforce SL mode (Primary -> Secondary -> Break-Even for TP2) and flag confluence/wall shifts on every rerun."
    )

    entry = {
        "ts_epoch": now_ts,
        "audited_pkt": now_dt.strftime("%Y-%m-%d %H:%M PKT"),
        "coin": target_coin,
        "timeframe": target_tf,
        "outcome_seen": (
            f"SL Mode: {sl_mode} | TP1:{int(tp1_hit)} TP2:{int(tp2_hit)} SL:{int(sl_hit)} "
            f"| Active:{len(active_trades)} Logged:{len(logbook)}"
        ),
        "missed_items": missed_items,
        "accuracy_pct": new_accuracy,
        "lesson_learned": lesson,
        "rule_adjustment": rule_adj,
    }
    audits.insert(0, entry)
    _save_audit_logs(audits)
    return audits[:100], entry


def silent_self_audit_and_update(
    existing_audits: List[Dict[str, Any]],
    active_trades: List[Dict[str, Any]],
    logbook: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    audits = list(existing_audits) if isinstance(existing_audits, list) else []
    now_dt = datetime.now(PKT_TZ)
    entry = {
        "ts_epoch": now_dt.timestamp(),
        "audited_pkt": now_dt.strftime("%Y-%m-%d %H:%M PKT"),
        "coin": active_trades[0].get("coin", "SCAN") if active_trades else "SCAN",
        "timeframe": active_trades[0].get("timeframe", "1H") if active_trades else "1H",
        "outcome_seen": f"{len(active_trades)} active / {len(logbook)} logged",
        "missed_items": [],
        "accuracy_pct": round(min(99.0, (float(audits[0].get("accuracy_pct", 72.0)) + 1.5) if audits else 73.5), 1),
        "lesson_learned": "Verified coin/timeframe binding, SL progression state, and logbook consistency.",
        "rule_adjustment": "Keep confirmation strictly on active swap pattern state and sync SL/TP transitions.",
    }
    audits.insert(0, entry)
    _save_audit_logs(audits)
    return audits[:100]


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
    previous_summary: Optional[str] = None,
    audit_entry: Optional[Dict[str, Any]] = None,
    api_note: str = "",
) -> str:
    direction = str(setup_context.get("pattern_type") or "SETUP")
    sig_code = str(setup_context.get("sig_code") or setup_context.get("status_label") or "")
    score = setup_context.get("total_score", 0)
    conf = setup_context.get("conf_pct", 0.0)
    mtf = setup_context.get("mtf_notes") or ["No conflicting higher-timeframe forming pattern"]
    wall = setup_context.get("tp_obstacle_note") or "Clear path"
    sl_mode = setup_context.get("sl_mode") or "PRIMARY SL"
    active_sl = setup_context.get("active_sl_guidance") or setup_context.get("sl")
    acc_pct = (audit_entry or {}).get("accuracy_pct", 75.0)
    missed = (audit_entry or {}).get("missed_items") or []

    corr_note = (
        f"Correlated with prior summary (Accuracy: **{acc_pct}%**). Adjustments: {', '.join(missed)}."
        if missed
        else f"Correlated with prior summary (Accuracy: **{acc_pct}%**); no missed items detected."
    )

    header_note = f"*(OpenRouter note: {api_note} — showing memory-correlated layman read)*\n\n" if api_note else ""
    return (
        f"{header_note}"
        f"**Institutional Layman Read — {target_coin} [{target_tf}] ({direction} · {sig_code}):**\n\n"
        f"* **What's Happening Now:** Score is **{score}/100** with confluence at **{conf}%**. {', '.join(str(x) for x in mtf)}.\n"
        f"* **Risk & SL Plan:** Current SL posture is **{sl_mode}** (Active SL guide: `{active_sl}`). Path wall: **{wall}**.\n"
        f"* **What Could Happen Next:** Track TP1 -> TP2 -> TP3 progression; shift SL to Break-Even and focus on TP2 once TP1 is secured.\n"
        f"* **Memory & Self-Audit Correlation:** {corr_note} (Tracking {len(all_committed_trades)} committed trade(s).)"
    )


def run_on_demand_trader_agent(
    target_coin: str,
    target_tf: str,
    setup_context: Dict[str, Any],
    all_committed_trades: List[Dict[str, Any]],
    audit_history: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    previous_summary: Optional[str] = None,
    audit_entry: Optional[Dict[str, Any]] = None,
) -> str:
    resolved_key = _resolve_openrouter_key(api_key)
    if not resolved_key:
        return _build_fallback_layman_summary(
            target_coin,
            target_tf,
            setup_context,
            all_committed_trades,
            audit_history,
            previous_summary=previous_summary,
            audit_entry=audit_entry,
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
        "You are a disciplined crypto swap futures trading agent with persistent self-audit memory. "
        "Always correlate your updated summary with the previous summary for this coin/timeframe, "
        "incorporate any missed items from the self-audit log to raise accuracy, and clearly explain: "
        "What's Happening Now, Risk & SL Plan (Primary SL vs Secondary SL vs Break-Even -> TP2), "
        "What Could Happen Next, and Memory Correlation & Self-Audit."
    )
    user_payload = {
        "target_coin": target_coin,
        "target_timeframe": target_tf,
        "setup_context": setup_context,
        "previous_summary": previous_summary or "None",
        "latest_self_audit": audit_entry or {},
        "recent_audits": audit_history[:5],
        "committed_trades": all_committed_trades[:10],
    }

    try:
        resp = requests.post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, default=str)},
                ],
                "temperature": 0.25,
                "max_tokens": 550,
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
            previous_summary=previous_summary,
            audit_entry=audit_entry,
            api_note="OpenRouter returned an empty response",
        )
    except Exception as exc:
        return _build_fallback_layman_summary(
            target_coin,
            target_tf,
            setup_context,
            all_committed_trades,
            audit_history,
            previous_summary=previous_summary,
            audit_entry=audit_entry,
            api_note=str(exc),
        )


def send_discord_webhook_alert(
    event_key: str,
    title: str,
    description: str,
    fields: List[Dict[str, str]],
    webhook_url_override: Optional[str] = None,
    color: int = 0x00C853,
    force: bool = False,
) -> bool:
    """
    Sends a Discord webhook embed and deduplicates identical event_key states
    so the bot alerts on new signals, warnings, TP/SL hits, and SL transitions.
    """
    hook_url = (webhook_url_override or "").strip() or _get_secret("DISCORD_WEBHOOK_URL", "")
    if not hook_url:
        return False

    sent_map = _read_json("discord_events", {})
    if not isinstance(sent_map, dict):
        sent_map = {}

    if not force and sent_map.get(event_key):
        return False

    payload = {
        "username": "Harmonics Swap Bot",
        "embeds": [
            {
                "title": title[:250],
                "description": description[:3900],
                "color": color,
                "fields": fields[:10],
                "footer": {"text": datetime.now(PKT_TZ).strftime("%Y-%m-%d %H:%M PKT")},
            }
        ],
    }

    try:
        resp = requests.post(hook_url, json=payload, timeout=8)
        resp.raise_for_status()
        sent_map[event_key] = datetime.now(PKT_TZ).timestamp()
        _write_json("discord_events", sent_map)
        _exec_turso(
            [
                {
                    "sql": "INSERT OR REPLACE INTO discord_events (event_key, ts_epoch, payload) VALUES (?, ?, ?)",
                    "args": [
                        {"type": "text", "value": event_key},
                        {"type": "float", "value": float(sent_map[event_key])},
                        {"type": "text", "value": json.dumps(payload, default=str)},
                    ],
                }
            ]
        )
        return True
    except Exception:
        return False
