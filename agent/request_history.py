"""
agent/request_history.py

Stores last N requests per session for "status of my last request" and observability.
Passed in/out via agent state; UI holds it in session_state.
Routing uses this memory (injected into the intent classifier) — no hardcoded phrase list.
"""

import time
from typing import Optional

# Max entries to keep (per session)
DEFAULT_MAX_ENTRIES = 20


def ensure_list(history: Optional[list]) -> list:
    """Return history as a list (copy to avoid mutating caller's)."""
    if history is None:
        return []
    return list(history)


def append_request(
    history: list,
    query: str,
    intent: str,
    success: bool,
    tool_name: str = "",
    error_type: str = "",
    summary: str = "",
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> list:
    """Append one request record and trim to max_entries. Returns new list."""
    entry = {
        "query": query,
        "intent": intent,
        "success": success,
        "tool_name": tool_name or "",
        "error_type": error_type or "",
        "summary": summary[:500] if summary else "",
        "timestamp": time.time(),
    }
    out = ensure_list(history) + [entry]
    return out[-max_entries:]


def format_last_requests(history: list, last_n: int = 5) -> str:
    """Format last N requests for display (e.g. in System Search)."""
    if not history:
        return "No previous requests in this session."
    entries = history[-last_n:]
    lines = []
    for i, e in enumerate(reversed(entries), 1):
        status = "✅" if e.get("success") else "❌"
        err = f" ({e.get('error_type', '')})" if e.get("error_type") else ""
        lines.append(
            f"{i}. {status} **{e.get('query', '')[:60]}** — {e.get('intent', '')} / {e.get('tool_name', '')}{err}"
        )
        if e.get("summary"):
            lines.append(f"   Summary: {e['summary'][:120]}...")
    return "**Last requests in this session:**\n\n" + "\n".join(lines)
