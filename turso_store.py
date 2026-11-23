import os
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


def _get_secret(key: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None:
                return str(val).strip()
    except Exception:
        pass
    return str(os.environ.get(key, default) or default).strip()


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
