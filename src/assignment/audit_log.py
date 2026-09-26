"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start an auditable request and return its stable request ID."""
        request_id = request_id or uuid.uuid4().hex
        from guardrails.output_guardrails import content_filter

        safe_input = content_filter(text)["redacted"]
        self._open[request_id] = {
            "started_monotonic": time.monotonic(),
            "user_id": user_id,
            "input": safe_input,
            "started_at": utc_now_iso(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close a request with a redacted output and measured latency."""
        request_id = request_id or uuid.uuid4().hex
        opened = self._open.pop(request_id, None)
        from guardrails.output_guardrails import content_filter

        safe_output = content_filter(text)["redacted"]
        now = time.monotonic()
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "input": (opened or {}).get("input", ""),
            "output": safe_output,
            "blocked": bool(blocked),
            "layer": layer,
            "started_at": (opened or {}).get("started_at"),
            "completed_at": utc_now_iso(),
            "latency_ms": round((now - opened["started_monotonic"]) * 1000, 3) if opened else None,
        })
        return self.logs[-1]

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
