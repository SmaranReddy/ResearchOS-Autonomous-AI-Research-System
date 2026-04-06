"""
Structured JSON request logger.

Appends one JSON line per request to logs/requests.jsonl (relative to the
backend root).  Log failures are printed and never propagated — a logging
bug must never take down a request.

Schema (all fields always present; optional fields set to null when absent):
  {
    "timestamp":    "2026-04-06T12:34:56.789Z",   # ISO-8601 UTC
    "query":        "What is RAG?",
    "status":       "success",                     # success | low_confidence | fallback | error
    "confidence":   0.823,
    "latency": {
      "retrieve_ms": 310,
      "rerank_ms":   150,
      "llm_ms":      620
    },
    "sources_count": 3,
    "cached":        false,
    "error":         null                          # error message string, or null
  }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Resolve log path relative to this file's parent's parent (backend/)
_LOG_DIR  = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "requests.jsonl"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_request(
    *,
    query: str,
    status: str,
    confidence: float,
    latency: dict,
    sources_count: int,
    cached: bool,
    error: Optional[str] = None,
) -> None:
    """
    Write a single log record.  Never raises — all exceptions are caught and
    printed so logging issues don't affect the request path.
    """
    try:
        _ensure_log_dir()
        record = {
            "timestamp":    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "query":        query,
            "status":       status,
            "confidence":   round(confidence, 4),
            "latency": {
                "retrieve_ms": latency.get("retrieve_ms", 0),
                "rerank_ms":   latency.get("rerank_ms",   0),
                "llm_ms":      latency.get("llm_ms",      0),
            },
            "sources_count": sources_count,
            "cached":        cached,
            "error":         error,
        }
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[LOGGER] failed to write log entry: {exc}")
