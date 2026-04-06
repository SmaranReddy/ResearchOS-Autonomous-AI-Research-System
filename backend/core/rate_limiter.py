"""
Per-IP rate limiter with a swappable backend.

Current implementation: in-memory sliding-window log.
Upgrade path: swap InMemoryRateLimiter for a Redis-backed implementation
              (e.g. using redis-py with a sorted-set per IP) by changing
              get_rate_limiter() to return the new class — no call-site changes needed.

Limitations of InMemoryRateLimiter
------------------------------------
- State is not shared across worker processes.
  Run with a single Uvicorn worker, or replace with Redis for multi-process deploys.
- State resets on every process restart — brief surge window possible after restart.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict

from fastapi import HTTPException, Request

_RATE_LIMIT  = int(os.getenv("RATE_LIMIT_RPM",   "20"))
_RATE_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds


# ---------------------------------------------------------------------------
# Abstract base — swap implementations without touching call sites
# ---------------------------------------------------------------------------

class AbstractRateLimiter(ABC):
    @abstractmethod
    def check(self, ip: str) -> None:
        """
        Check whether *ip* is within the allowed request rate.
        Raises HTTPException(429) if the limit is exceeded.
        Records the current request on success.
        """


# ---------------------------------------------------------------------------
# In-memory sliding-window implementation
# ---------------------------------------------------------------------------

class InMemoryRateLimiter(AbstractRateLimiter):
    """
    Tracks per-IP timestamps in a dict[str, list[float]].
    Thread-safe for single-worker FastAPI (no concurrent async writes to the same list).
    """

    def __init__(self, limit: int = _RATE_LIMIT, window: int = _RATE_WINDOW) -> None:
        self._limit  = limit
        self._window = window
        self._log: dict[str, list[float]] = defaultdict(list)

    def check(self, ip: str) -> None:
        now = time.monotonic()

        # Evict timestamps outside the window
        self._log[ip] = [t for t in self._log[ip] if now - t < self._window]

        if len(self._log[ip]) >= self._limit:
            retry_after = int(self._window - (now - self._log[ip][0])) + 1
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded: {self._limit} requests per {self._window}s. "
                    f"Retry after {retry_after}s."
                ),
            )

        self._log[ip].append(now)


# ---------------------------------------------------------------------------
# Module-level singleton + FastAPI dependency
# ---------------------------------------------------------------------------

_limiter: AbstractRateLimiter = InMemoryRateLimiter()


def get_rate_limiter() -> AbstractRateLimiter:
    """Return the active rate-limiter instance (replace body to swap backends)."""
    return _limiter


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """FastAPI dependency — use with Depends(rate_limit)."""
    get_rate_limiter().check(_get_client_ip(request))
