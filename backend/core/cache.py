"""
In-memory response cache with TTL, LRU eviction, and index-version invalidation.

Key  : SHA-256 of (normalised query + history + index_version)
Value: full pipeline result (QueryResponse object)
TTL  : 10 minutes  (CACHE_TTL_SECONDS env var)
Size : 100 entries (CACHE_MAX_ENTRIES env var)

Cache invalidation strategy
----------------------------
index_version is a process-level integer incremented every time new documents
are written to Pinecone (in executor._step_index).  Including the version in
the cache key means that any response cached before ingestion will never be
returned after ingestion — the old keys simply become unreachable and expire
naturally via LRU eviction or TTL.

Only "success" responses are stored.  "fallback", "low_confidence", and
"error" are never cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Any

_TTL:     int = int(os.getenv("CACHE_TTL_SECONDS", "600"))   # 10 min
_MAXSIZE: int = int(os.getenv("CACHE_MAX_ENTRIES", "100"))

# ---------------------------------------------------------------------------
# Index-version counter
# Incremented by executor._step_index whenever at least one paper is indexed.
# ---------------------------------------------------------------------------
_index_version: int = 0


def get_index_version() -> int:
    return _index_version


def increment_index_version() -> None:
    global _index_version
    _index_version += 1
    print(f"[CACHE] index_version incremented → {_index_version}  (all cached responses invalidated)")


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------

class ResponseCache:
    """Thread-safe for single-worker FastAPI (no concurrent async writes)."""

    def __init__(self, ttl: int = _TTL, maxsize: int = _MAXSIZE) -> None:
        self._ttl     = ttl
        self._maxsize = maxsize
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def _key(self, query: str, history: list) -> str:
        """
        Key includes index_version so that post-ingestion queries never
        receive pre-ingestion cached answers.
        """
        payload = (
            query.strip().lower()
            + "||"
            + json.dumps(history, sort_keys=True, ensure_ascii=False)
            + "||v"
            + str(get_index_version())
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, query: str, history: list) -> Any | None:
        key = self._key(query, history)
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            print("[CACHE] expired entry evicted")
            return None
        self._store.move_to_end(key)
        print(f"[CACHE] HIT  version={get_index_version()}  size={len(self._store)}")
        return value

    def set(self, query: str, history: list, value: Any) -> None:
        key = self._key(query, history)
        self._store[key] = (time.monotonic(), value)
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)
            print(f"[CACHE] LRU eviction  maxsize={self._maxsize}")
        print(f"[CACHE] SET  version={get_index_version()}  size={len(self._store)}")

    def __len__(self) -> int:
        return len(self._store)


# Module-level singleton
_cache = ResponseCache()


def get_cache() -> ResponseCache:
    return _cache
