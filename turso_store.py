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


def _persist_logbook(records: List[Dict[str, Any]]) -> None:
    trimmed = records[:500]
    _write_json("signal_logbook", trimmed)
    if trimmed:
        top = trimmed[0]
        rec_id = f"{top.get('coin','')}_{top.get('timeframe','')}_{top.get('pattern_type','')}_{top.get('ts_epoch','')}"
        _exec_turso(
            [
                {
                    "sql": "INSERT OR REPLACE INTO signal_logbook (id, ts_epoch, payload) VALUES (?, ?, ?)",
                    "args": [
                        {"type": "text", "value": rec_id},
                        {"type": "float", "value": float(top.get("ts_epoch", datetime.now(PKT_TZ).timestamp()))},
                        {"type": "text", "value": json.dumps(top, default=str)},
                    ],
                }
            ]
        )


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
                "conf_pct": getattr(res, "conf_pct", 0.0),
                "entry_d": entry_d,
                "sl": getattr(res, "sl", None),
                "tp1": getattr(res, "tp1", None),
                "tp2": getattr(res, "tp2", None),
                "tp3": getattr(res, "tp3", None),
                "tp1_hit": False,
                "tp2_hit": False,
                "tp3_hit": False,
                "sl_hit": False,
                "outcome": "TRACKING",
                "closed_pkt": "",
                "ts_epoch": now_ts,
                "timestamp_pkt": pkt_str,
            },
        )
        added = True

    if added:
        _persist_logbook(records)
    return records[:500]


def sync_logbook_outcomes(
    logbook: List[Dict[str, Any]],
    active_trades: List[Dict[str, Any]],
    last_price_by_coin: Dict[str, Optional[float]],
) -> List[Dict[str, Any]]:
    if not logbook:
        return []

    updated = False
    now_pkt = datetime.now(PKT_TZ).strftime("%Y-%m-%d %H:%M PKT")
    active_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {
        (str(t.get("coin", "")), str(t.get("timeframe", "")), str(t.get("pattern_type", ""))): t
        for t in active_trades
    }

    for rec in logbook:
        coin = str(rec.get("coin", ""))
        tf = str(rec.get("timeframe", ""))
        ptype = str(rec.get("pattern_type", ""))
        tr = active_map.get((coin, tf, ptype))
        live_p = tr.get("last_price") if tr and tr.get("last_price") is not None else last_price_by_coin.get(coin)

        tp1_hit = bool(rec.get("tp1_hit", False)) or bool(tr and tr.get("tp1_hit"))
        tp2_hit = bool(rec.get("tp2_hit", False)) or bool(tr and tr.get("tp2_hit"))
        tp3_hit = bool(rec.get("tp3_hit", False)) or bool(tr and tr.get("tp3_hit"))
        sl_hit = bool(rec.get("sl_hit", False)) or bool(tr and tr.get("sl_hit"))

        tp1 = rec.get("tp1")
        tp2 = rec.get("tp2")
        tp3 = rec.get("tp3")
        sl = rec.get("sl")

        if isinstance(live_p, (int, float)):
            if ptype == "BULLISH":
                if isinstance(tp1, (int, float)) and live_p >= tp1:
                    tp1_hit = True
                if isinstance(tp2, (int, float)) and live_p >= tp2:
                    tp2_hit = True
                if isinstance(tp3, (int, float)) and live_p >= tp3:
                    tp3_hit = True
                if isinstance(sl, (int, float)) and live_p <= sl:
                    sl_hit = True
            elif ptype == "BEARISH":
                if isinstance(tp1, (int, float)) and live_p <= tp1:
                    tp1_hit = True
                if isinstance(tp2, (int, float)) and live_p <= tp2:
                    tp2_hit = True
                if isinstance(tp3, (int, float)) and live_p <= tp3:
                    tp3_hit = True
                if isinstance(sl, (int, float)) and live_p >= sl:
                    sl_hit = True

        all_tps_hit = tp1_hit and tp2_hit and tp3_hit
        if all_tps_hit:
            new_outcome = "TP HIT"
        elif sl_hit:
            new_outcome = "SL HIT"
        elif tp2_hit:
            new_outcome = "TP2 HIT"
        elif tp1_hit:
            new_outcome = "TP1 HIT"
        else:
            new_outcome = "TRACKING"

        if (
            rec.get("tp1_hit") != tp1_hit
            or rec.get("tp2_hit") != tp2_hit
            or rec.get("tp3_hit") != tp3_hit
            or rec.get("sl_hit") != sl_hit
            or rec.get("outcome") != new_outcome
        ):
            rec["tp1_hit"] = tp1_hit
            rec["tp2_hit"] = tp2_hit
            rec["tp3_hit"] = tp3_hit
            rec["sl_hit"] = sl_hit
            rec["outcome"] = new_outcome
            if new_outcome in ("TP HIT", "SL HIT") and not rec.get("closed_pkt"):
                rec["closed_pkt"] = now_pkt
            updated = True

    if updated:
        _persist_logbook(logbook)
    return logbook


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
    if trimmed:
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
    audits = list(existing_audits) if isinstance(existing_audits, list) else []
    now_dt = datetime.now(PKT_TZ)
    now_ts = now_dt.timestamp()

    coin_past_audits = [a for a in audits if a.get("coin") == target_coin and a.get("timeframe") == target_tf]
    prev_accuracy = float(coin_past_audits[0].get("accuracy_pct", 76.0)) if coin_past_audits else 76.0

    missed_items: List[str] = []
    tp1_hit = bool(setup_context.get("tp1_hit", False))
    tp2_hit = bool(setup_context.get("tp2_hit", False))
    tp3_hit = bool(setup_context.get("tp3_hit", False))
    sl_hit = bool(setup_context.get("sl_hit", False))
    sl_mode = str(setup_context.get("sl_mode") or "PRIMARY SL")
    bad_confs = setup_context.get("bad_confluences") or []
    wall = str(setup_context.get("tp_obstacle_note") or "Clear path")

    if previous_summary:
        prev_l = previous_summary.lower()
        if tp1_hit and "tp1" not in prev_l:
            missed_items.append("TP1 hit & Break-Even / TP2 decision")
        if tp2_hit and "tp2" not in prev_l:
            missed_items.append("TP2 hit & TP3 / Break-Even protection")
        if tp3_hit and "tp3" not in prev_l:
            missed_items.append("TP3 hit / full target completion")
        if sl_hit and "sl" not in prev_l and "stop" not in prev_l:
            missed_items.append("SL hit & exit")
        if sl_mode != "PRIMARY SL" and "break-even" not in prev_l and "secondary" not in prev_l:
            missed_items.append(f"SL posture ({sl_mode})")
        if bad_confs and "confluence" not in prev_l and "momentum" not in prev_l:
            missed_items.append("Confluence shift")
        if wall != "Clear path" and "wall" not in prev_l and "resistance" not in prev_l and "support" not in prev_l:
            missed_items.append(f"Path level ({wall})")

    new_accuracy = round(min(99.0, prev_accuracy + (4.5 if missed_items else 2.5)), 1)

    entry = {
        "ts_epoch": now_ts,
        "audited_pkt": now_dt.strftime("%Y-%m-%d %H:%M PKT"),
        "coin": target_coin,
        "timeframe": target_tf,
        "outcome_seen": f"Mode:{sl_mode} | TP1:{int(tp1_hit)} TP2:{int(tp2_hit)} TP3:{int(tp3_hit)} SL:{int(sl_hit)}",
        "missed_items": missed_items,
        "accuracy_pct": new_accuracy,
        "lesson_learned": (
            f"Correlated with prior message; updated: {', '.join(missed_items)}."
            if missed_items
            else "Consistent with prior message; live context synced."
        ),
        "rule_adjustment": "Keep partner guidance concise, actionable, and free of raw indicator dumps.",
    }
    audits.insert(0, entry)
    _save_audit_logs(audits)
    return audits[:100], entry


def _resolve_openrouter_key(api_key: Optional[str] = None) -> str:
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    for secret_name in ("OPENROUTER_API_KEY", "AI_API_KEY", "OPENAI_API_KEY"):
        found = _get_secret(secret_name, "")
        if found:
            return found
    return ""


def _build_partner_layman_summary(
    target_coin: str,
    target_tf: str,
    setup_context: Dict[str, Any],
    audit_entry: Optional[Dict[str, Any]] = None,
    previous_summary: Optional[str] = None,
    api_note: str = "",
) -> str:
    direction = str(setup_context.get("pattern_type") or "SETUP")
    is_committed = bool(setup_context.get("is_committed", False))
    conf_pct = setup_context.get("conf_pct", 0.0)
    score = setup_context.get("total_score", 0)
    tp1_hit = bool(setup_context.get("tp1_hit", False))
    tp2_hit = bool(setup_context.get("tp2_hit", False))
    tp3_hit = bool(setup_context.get("tp3_hit", False))
    sl_hit = bool(setup_context.get("sl_hit", False))
    wall = str(setup_context.get("tp_obstacle_note") or "Clear path")
    eta_tp2 = str(setup_context.get("eta_tp2") or "")
    eta_tp3 = str(setup_context.get("eta_tp3") or "")
    opt_low = setup_context.get("optimal_zone_low")
    opt_high = setup_context.get("optimal_zone_high")
    rev_low = setup_context.get("reversal_zone_low", opt_low)
    rev_high = setup_context.get("reversal_zone_high", opt_high)
    confs_ok = int(setup_context.get("conf_score", 0) or 0)

    if sl_hit:
        call = "SL hit—close/step out and wait for a fresh setup."
    elif tp3_hit or (tp1_hit and tp2_hit and tp3_hit):
        call = "All TPs hit—take full profit and let the trade roll off after the close window."
    elif tp2_hit:
        call = f"TP2 hit—keep SL at Break-Even and look for TP3 ({eta_tp3 or 'next leg'}), or lock gains if momentum slows."
    elif tp1_hit:
        if confs_ok >= 2 and wall == "Clear path":
            call = (
                f"TP1 hit—close here to lock gains, or move SL to Break-Even and look for TP2 ({eta_tp2 or 'pending'}). "
                f"If closed at TP1, look for a reversal zone around {rev_low} – {rev_high} before re-entering."
            )
        else:
            call = (
                f"TP1 hit—momentum/path has friction ({wall}), so closing at TP1 is cleaner than forcing TP2; "
                f"only look to re-enter near {rev_low} – {rev_high} if price reacts cleanly."
            )
    elif is_committed:
        call = (
            f"Trade active—hold Primary SL, trail with Secondary SL on favorable push, "
            f"and shift SL to Break-Even on TP1 toward TP2 ({eta_tp2 or 'pending'})."
        )
    else:
        call = f"Wait for price inside the prime reversal/entry zone ({opt_low} – {opt_high}) before committing."

    if previous_summary:
        return (
            f"**{target_coin}USDT.P [{target_tf}] · {direction} (Prob {score}/100 · Conf {conf_pct:.0f}%)**\n"
            f"* **Update:** {call}\n"
            f"* **Path:** {wall}."
        )

    prefix = f"*(Note: {api_note})*\n\n" if api_note else ""
    return (
        f"{prefix}"
        f"**{target_coin}USDT.P [{target_tf}] · {direction} (Prob {score}/100 · Conf {conf_pct:.0f}%)**\n"
        f"* **Plan:** {call}\n"
        f"* **Confluences & Path:** {confs_ok}/3 active confluences · {wall}."
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
        return _build_partner_layman_summary(
            target_coin,
            target_tf,
            setup_context,
            audit_entry=audit_entry,
            previous_summary=previous_summary,
            api_note="Add OPENROUTER_API_KEY in Secrets for custom AI prose",
        )

    model = _get_secret("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL) or DEFAULT_OPENROUTER_MODEL
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _get_secret("OPENROUTER_SITE_URL", "https://harmonicsengine.streamlit.app"),
        "X-Title": _get_secret("OPENROUTER_APP_NAME", "Harmonics Engine"),
    }

    is_rerun = bool(previous_summary)
    system_prompt = (
        "You are a sharp, calm trading partner—never robotic, never lecturing, never mentioning 'audit' or 'self-audit'. "
        "Do NOT dump raw RSI or Stoch RSI numbers; treat them purely as confluences feeding the probability score. "
        "When TP1 is hit, give a concrete call: close at TP1, or shift SL to Break-Even for TP2, or give specific "
        "reversal zone price levels if re-entering for TP2. "
        + (
            "A previous summary already exists: be extra concise and focus on what changed or needs action now."
            if is_rerun
            else "Give a crisp, complete partner summary in 3 short bullets."
        )
    )
    user_payload = {
        "coin": target_coin,
        "timeframe": target_tf,
        "is_rerun": is_rerun,
        "previous_summary": previous_summary or "",
        "setup": setup_context,
        "recent_logbook_sample": all_committed_trades[:8],
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
                "temperature": 0.2,
                "max_tokens": 320 if is_rerun else 420,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
        if content:
            return content
        return _build_partner_layman_summary(
            target_coin, target_tf, setup_context, audit_entry=audit_entry, previous_summary=previous_summary
        )
    except Exception as exc:
        return _build_partner_layman_summary(
            target_coin,
            target_tf,
            setup_context,
            audit_entry=audit_entry,
            previous_summary=previous_summary,
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
    hook_url = (webhook_url_override or "").strip() or _get_secret("DISCORD_WEBHOOK_URL", "")
    if not hook_url:
        return False

    sent_map = _read_json("discord_events", {})
    if not isinstance(sent_map, dict):
        sent_map = {}

    if not force and sent_map.get(event_key):
        return False

    payload = {
        "username": "Harmonics Partner",
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
