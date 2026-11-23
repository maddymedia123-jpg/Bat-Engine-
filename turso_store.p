from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
import streamlit as st

from market_data import now_pkt

LOCAL_FALLBACK = Path("harmonics_local_store.json")
HORIZONS = ["All Time", "1D", "1W", "1M"]


def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass
    import os

    return os.environ.get(key, default).strip()


def _turso_http_url() -> str:
    raw = _get_secret("TURSO_DATABASE_URL", "")
    if not raw:
        return ""
    if raw.startswith("libsql://"):
        base = "https://" + raw[len("libsql://") :]
    elif raw.startswith("turso://"):
        base = "https://" + raw[len("turso://") :]
    else:
        base = raw
    base = base.rstrip("/")
    return base if base.endswith("/v2/pipeline") else f"{base}/v2/pipeline"


def _turso_token() -> str:
    return _get_secret("TURSO_AUTH_TOKEN", "")


def turso_configured() -> bool:
    return bool(_turso_http_url() and _turso_token())


def _encode_arg(val: Any) -> Dict[str, Any]:
    if val is None:
        return {"type": "null"}
    if isinstance(val, bool):
        return {"type": "integer", "value": "1" if val else "0"}
    if isinstance(val, int):
        return {"type": "integer", "value": str(val)}
    if isinstance(val, float):
        return {"type": "float", "value": str(val)}
    return {"type": "text", "value": str(val)}


def exec_sql(sql: str, args: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
    url = _turso_http_url()
    token = _turso_token()
    if not url or not token:
        return None

    stmt: Dict[str, Any] = {"sql": sql}
    if args is not None:
        stmt["args"] = [_encode_arg(a) for a in args]

    payload = {
        "requests": [
            {"type": "execute", "stmt": stmt},
            {"type": "close"},
        ]
    }
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def init_storage() -> None:
    if not turso_configured():
        return
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS active_trades (
            trade_key TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            updated_pkt TEXT NOT NULL
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS signal_logbook (
            sig_key TEXT PRIMARY KEY,
            created_iso TEXT NOT NULL,
            timestamp_pkt TEXT NOT NULL,
            coin TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            status_label TEXT NOT NULL,
            sig_code TEXT NOT NULL,
            conf_pct REAL NOT NULL,
            total_score INTEGER NOT NULL,
            entry_d REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            tp3 REAL,
            outcome TEXT NOT NULL
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ai_summaries (
            context_key TEXT PRIMARY KEY,
            coin TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            summary_md TEXT NOT NULL,
            updated_pkt TEXT NOT NULL
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS agent_self_audit (
            audit_key TEXT PRIMARY KEY,
            coin TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            outcome_seen TEXT NOT NULL,
            lesson_learned TEXT NOT NULL,
            rule_adjustment TEXT NOT NULL,
            audited_pkt TEXT NOT NULL
        )
        """
    )


def _read_local_bundle() -> Dict[str, Any]:
    if not LOCAL_FALLBACK.exists():
        return {"active_trades": [], "logbook": [], "summaries": {}, "audits": []}
    try:
        data = json.loads(LOCAL_FALLBACK.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"active_trades": [], "logbook": [], "summaries": {}, "audits": []}
    except Exception:
        return {"active_trades": [], "logbook": [], "summaries": {}, "audits": []}


def _write_local_bundle(bundle: Dict[str, Any]) -> None:
    try:
        LOCAL_FALLBACK.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_active_trades() -> List[Dict[str, Any]]:
    if turso_configured():
        res = exec_sql("SELECT payload_json FROM active_trades")
        if res:
            try:
                rows = res["results"][0]["response"]["result"]["rows"]
                out: List[Dict[str, Any]] = []
                for row in rows:
                    raw_val = row[0].get("value")
                    if raw_val:
                        parsed = json.loads(raw_val)
                        if isinstance(parsed, dict):
                            out.append(parsed)
                return out
            except Exception:
                pass
    bundle = _read_local_bundle()
    return [t for t in bundle.get("active_trades", []) if isinstance(t, dict)]


def save_active_trades(trades: List[Dict[str, Any]]) -> None:
    clean = [t for t in trades if isinstance(t, dict)]
    bundle = _read_local_bundle()
    bundle["active_trades"] = clean
    _write_local_bundle(bundle)

    if not turso_configured():
        return
    exec_sql("DELETE FROM active_trades")
    ts = now_pkt().strftime("%Y-%m-%d %H:%M:%S PKT")
    for t in clean:
        key = f"{t.get('coin')}_{t.get('timeframe')}_{t.get('pattern_type')}"
        exec_sql(
            "INSERT OR REPLACE INTO active_trades (trade_key, payload_json, updated_pkt) VALUES (?, ?, ?)",
            [key, json.dumps(t), ts],
        )


def load_logbook() -> List[Dict[str, Any]]:
    if turso_configured():
        res = exec_sql(
            """
            SELECT created_iso, timestamp_pkt, coin, timeframe, pattern_type,
                   status_label, sig_code, conf_pct, total_score,
                   entry_d, sl, tp1, tp2, tp3, outcome
            FROM signal_logbook
            ORDER BY created_iso DESC
            LIMIT 500
            """
        )
        if res:
            try:
                rows = res["results"][0]["response"]["result"]["rows"]
                records: List[Dict[str, Any]] = []
                for r in rows:
                    def _v(idx: int) -> Any:
                        cell = r[idx]
                        return None if cell.get("type") == "null" else cell.get("value")

                    records.append(
                        {
                            "created_iso": _v(0) or "",
                            "timestamp_pkt": _v(1) or "",
                            "coin": _v(2) or "",
                            "timeframe": _v(3) or "",
                            "pattern_type": _v(4) or "",
                            "status_label": _v(5) or "",
                            "sig_code": _v(6) or "-",
                            "conf_pct": float(_v(7) or 0.0),
                            "total_score": int(float(_v(8) or 0)),
                            "entry_d": float(_v(9)) if _v(9) is not None else None,
                            "sl": float(_v(10)) if _v(10) is not None else None,
                            "tp1": float(_v(11)) if _v(11) is not None else None,
                            "tp2": float(_v(12)) if _v(12) is not None else None,
                            "tp3": float(_v(13)) if _v(13) is not None else None,
                            "outcome": _v(14) or "TRACKING",
                        }
                    )
                if records:
                    return records
            except Exception:
                pass

    bundle = _read_local_bundle()
    return [r for r in bundle.get("logbook", []) if isinstance(r, dict)]


def record_generated_signals(records: List[Any], scan_results: List[Any]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = [r for r in records if isinstance(r, dict)]
    existing_keys = {
        (
            r.get("coin"),
            r.get("timeframe"),
            r.get("pattern_type"),
            r.get("status_label"),
            round(float(r.get("entry_d") or 0.0), 6),
        )
        for r in valid
    }
    ts_str = now_pkt().strftime("%a %H:%M PKT")
    iso_str = now_pkt().replace(tzinfo=None).isoformat()
    newly_added: List[Dict[str, Any]] = []

    for res in scan_results:
        status_label = getattr(res, "status_label", "IDLE")
        entry_d = getattr(res, "entry_d", None)
        if status_label not in ("READY TO BE", "CONFIRMED") or entry_d is None:
            continue
        coin = getattr(res, "coin", "")
        timeframe = getattr(res, "timeframe", "")
        pattern_type = getattr(res, "pattern_type", "")
        key = (coin, timeframe, pattern_type, status_label, round(float(entry_d), 6))
        if key not in existing_keys:
            existing_keys.add(key)
            rec = {
                "created_iso": iso_str,
                "timestamp_pkt": ts_str,
                "coin": coin,
                "timeframe": timeframe,
                "pattern_type": pattern_type,
                "status_label": status_label,
                "sig_code": getattr(res, "sig_code", "-"),
                "conf_pct": float(getattr(res, "conf_pct", 0.0)),
                "total_score": int(getattr(res, "total_score", 0)),
                "entry_d": float(entry_d),
                "sl": getattr(res, "sl", None),
                "tp1": getattr(res, "tp1", None),
                "tp2": getattr(res, "tp2", None),
                "tp3": getattr(res, "tp3", None),
                "outcome": "TRACKING",
            }
            valid.insert(0, rec)
            newly_added.append(rec)

    if newly_added:
        bundle = _read_local_bundle()
        bundle["logbook"] = valid
        _write_local_bundle(bundle)

        if turso_configured():
            for rec in newly_added:
                sig_key = f"{rec['coin']}_{rec['timeframe']}_{rec['pattern_type']}_{rec['status_label']}_{round(rec['entry_d'], 6)}"
                exec_sql(
                    """
                    INSERT OR IGNORE INTO signal_logbook (
                        sig_key, created_iso, timestamp_pkt, coin, timeframe, pattern_type,
                        status_label, sig_code, conf_pct, total_score,
                        entry_d, sl, tp1, tp2, tp3, outcome
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        sig_key,
                        rec["created_iso"],
                        rec["timestamp_pkt"],
                        rec["coin"],
                        rec["timeframe"],
                        rec["pattern_type"],
                        rec["status_label"],
                        rec["sig_code"],
                        rec["conf_pct"],
                        rec["total_score"],
                        rec["entry_d"],
                        rec["sl"],
                        rec["tp1"],
                        rec["tp2"],
                        rec["tp3"],
                        rec["outcome"],
                    ],
                )

    return valid


def filter_logbook_records(records: List[Any], horizon: str) -> List[Dict[str, Any]]:
    valid = [r for r in records if isinstance(r, dict)]
    if not valid or horizon == "All Time":
        return valid

    cutoff = now_pkt().replace(tzinfo=None) - {
        "1D": pd.Timedelta(days=1),
        "1W": pd.Timedelta(weeks=1),
        "1M": pd.Timedelta(days=30),
    }.get(horizon, pd.Timedelta(days=1))

    out: List[Dict[str, Any]] = []
    for r in valid:
        dt = pd.to_datetime(r.get("created_iso"), errors="coerce")
        if pd.notna(dt) and dt >= cutoff:
            out.append(r)
    return out if out else valid[:25]


def load_saved_summaries() -> Dict[str, str]:
    if turso_configured():
        res = exec_sql("SELECT context_key, summary_md FROM ai_summaries")
        if res:
            try:
                rows = res["results"][0]["response"]["result"]["rows"]
                return {
                    str(r[0].get("value")): str(r[1].get("value"))
                    for r in rows
                    if r[0].get("value") and r[1].get("value")
                }
            except Exception:
                pass
    bundle = _read_local_bundle()
    return bundle.get("summaries", {}) if isinstance(bundle.get("summaries"), dict) else {}


def save_summary(context_key: str, coin: str, timeframe: str, summary_md: str) -> None:
    bundle = _read_local_bundle()
    summaries = bundle.get("summaries") if isinstance(bundle.get("summaries"), dict) else {}
    summaries[context_key] = summary_md
    bundle["summaries"] = summaries
    _write_local_bundle(bundle)

    if turso_configured():
        ts = now_pkt().strftime("%Y-%m-%d %H:%M:%S PKT")
        exec_sql(
            "INSERT OR REPLACE INTO ai_summaries (context_key, coin, timeframe, summary_md, updated_pkt) VALUES (?, ?, ?, ?, ?)",
            [context_key, coin, timeframe, summary_md, ts],
        )


def load_audit_logs() -> List[Dict[str, Any]]:
    if turso_configured():
        res = exec_sql(
            "SELECT coin, timeframe, outcome_seen, lesson_learned, rule_adjustment, audited_pkt FROM agent_self_audit ORDER BY audited_pkt DESC LIMIT 100"
        )
        if res:
            try:
                rows = res["results"][0]["response"]["result"]["rows"]
                return [
                    {
                        "coin": r[0].get("value", ""),
                        "timeframe": r[1].get("value", ""),
                        "outcome_seen": r[2].get("value", ""),
                        "lesson_learned": r[3].get("value", ""),
                        "rule_adjustment": r[4].get("value", ""),
                        "audited_pkt": r[5].get("value", ""),
                    }
                    for r in rows
                ]
            except Exception:
                pass
    bundle = _read_local_bundle()
    return [a for a in bundle.get("audits", []) if isinstance(a, dict)]


def silent_self_audit_and_update(
    existing_audits: List[Dict[str, Any]],
    active_trades: List[Dict[str, Any]],
    logbook: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Silently audits committed trades and logged signals whenever a user runs or re-runs
    an AI summary, updating the agent's memory and rule adjustments without cluttering UI.
    """
    audits = [a for a in existing_audits if isinstance(a, dict)]
    seen_keys = {f"{a.get('coin')}_{a.get('timeframe')}_{a.get('outcome_seen')}" for a in audits}
    ts = now_pkt().strftime("%Y-%m-%d %H:%M:%S PKT")
    added = False

    for t in active_trades:
        coin = t.get("coin", "")
        tf = t.get("timeframe", "")
        sl_hit = bool(t.get("sl_hit", False))
        tp1_hit = bool(t.get("tp1_hit", False))
        if sl_hit and not tp1_hit:
            outcome_tag = "SL_BEFORE_TP1"
            lesson = "Wick/price tagged primary ATR SL before TP1; premature or over-extended entry may have faced liquidity sweep."
            rule_adj = "Require closer proximity to optimal reversal zone & confirm momentum turn before committing."
        elif sl_hit and tp1_hit:
            outcome_tag = "TP1_THEN_SL"
            lesson = "TP1 reached before reversal back to SL; locking partial/moving secondary SL after TP1 protects gains."
            rule_adj = "Prioritize moving secondary SL once TP1 is hit."
        elif tp1_hit:
            outcome_tag = "TP_PROGRESS"
            lesson = "Trade progressed toward target levels as projected."
            rule_adj = "Maintain confluence alignment and watch higher-timeframe reaction zones."
        else:
            outcome_tag = "ACTIVE_OPEN"
            lesson = "Trade currently monitored; evaluate HTF pull/opposition and wall proximity."
            rule_adj = "Penalize setups with immediate support/resistance walls ahead of TP1."

        akey = f"{coin}_{tf}_{outcome_tag}"
        if akey not in seen_keys:
            seen_keys.add(akey)
            entry = {
                "coin": coin,
                "timeframe": tf,
                "outcome_seen": outcome_tag,
                "lesson_learned": lesson,
                "rule_adjustment": rule_adj,
                "audited_pkt": ts,
            }
            audits.insert(0, entry)
            added = True
            if turso_configured():
                exec_sql(
                    "INSERT OR REPLACE INTO agent_self_audit (audit_key, coin, timeframe, outcome_seen, lesson_learned, rule_adjustment, audited_pkt) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [akey, coin, tf, outcome_tag, lesson, rule_adj, ts],
                )

    for rec in logbook[:20]:
        if "SL" in str(rec.get("outcome", "")).upper():
            coin = rec.get("coin", "")
            tf = rec.get("timeframe", "")
            akey = f"{coin}_{tf}_LOG_SL"
            if akey not in seen_keys:
                seen_keys.add(akey)
                entry = {
                    "coin": coin,
                    "timeframe": tf,
                    "outcome_seen": "LOG_SL",
                    "lesson_learned": f"Logged signal on {coin} {tf} hit SL; checking score/confluence sensitivity.",
                    "rule_adjustment": "Weight HTF opposition and volatility spikes more strictly.",
                    "audited_pkt": ts,
                }
                audits.insert(0, entry)
                added = True
                if turso_configured():
                    exec_sql(
                        "INSERT OR REPLACE INTO agent_self_audit (audit_key, coin, timeframe, outcome_seen, lesson_learned, rule_adjustment, audited_pkt) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [akey, coin, tf, "LOG_SL", entry["lesson_learned"], entry["rule_adjustment"], ts],
                    )

    if added:
        bundle = _read_local_bundle()
        bundle["audits"] = audits
        _write_local_bundle(bundle)

    return audits


def run_on_demand_trader_agent(
    target_coin: str,
    target_tf: str,
    setup_context: Dict[str, Any],
    all_committed_trades: List[Dict[str, Any]],
    audit_history: List[Dict[str, Any]],
    api_key: str = "",
) -> str:
    """
    Runs ONLY when explicitly clicked by the user. Acts as an institutional harmonic
    trader agent, correlates with committed trades & self-audit lessons, and explains
    what could happen next in plain layman language.
    """
    resolved_key = api_key.strip() or _get_secret("OPENAI_API_KEY", "")
    recent_lessons = [
        f"{a.get('coin')} ({a.get('timeframe')}): {a.get('lesson_learned')} -> {a.get('rule_adjustment')}"
        for a in audit_history[:5]
    ]
    committed_brief = [
        f"{t.get('coin')} [{t.get('timeframe')}] {t.get('pattern_type')} Entry={t.get('entry_d')} SL={t.get('sl')} TP1={t.get('tp1')}"
        for t in all_committed_trades
    ]

    if resolved_key:
        try:
            prompt_payload = {
                "target": {"coin": target_coin, "timeframe": target_tf, "setup": setup_context},
                "all_committed_trades": committed_brief,
                "self_audit_memory": recent_lessons,
            }
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an institutional harmonic trading specialist and risk manager. "
                                "Explain the trade in simple, clear layman language: (1) current structure & "
                                "confluence health, (2) how committed trades and past self-audit lessons apply, "
                                "(3) what could happen next around the reversal zone, SL, and TPs."
                            ),
                        },
                        {"role": "user", "content": json.dumps(prompt_payload)},
                    ],
                    "temperature": 0.25,
                },
                timeout=14,
            )
            resp.raise_for_status()
            txt = resp.json()["choices"][0]["message"]["content"]
            if txt:
                return txt.strip()
        except Exception as exc:
            prefix = f"*(AI API note: {exc} — showing structured institutional layman read)*\n\n"
        else:
            prefix = ""
    else:
        prefix = ""

    notes_str = " ".join(setup_context.get("mtf_notes", [])) if isinstance(setup_context.get("mtf_notes"), list) else ""
    lesson_str = recent_lessons[0] if recent_lessons else "No prior SL/TP mistakes recorded yet."
    return (
        f"{prefix}"
        f"**Institutional Layman Read — {target_coin} [{target_tf}] ({setup_context.get('pattern_type', '')} · {setup_context.get('status_label', '')}):**\n"
        f"* **What's Happening Now:** Score is **{setup_context.get('total_score', 0)}/100** with confluence at **{setup_context.get('conf_pct', 0)}%**. {notes_str}\n"
        f"* **Risk & SL Plan:** Primary ATR SL is the main line in the sand; if price moves in your favor, a secondary SL can trail behind structure/TP1 without rushing.\n"
        f"* **What Could Happen Next:** Watch price reaction around the reversal/optimal entry zone and nearby support/resistance walls before expecting continuation toward TP1–TP3.\n"
        f"* **Agent Memory Note:** {lesson_str}"
    )
