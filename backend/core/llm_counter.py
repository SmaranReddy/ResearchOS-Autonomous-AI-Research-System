"""
Thread-local LLM call counter.

Each thread (request) gets its own counter so concurrent requests don't
pollute each other's totals.  Call reset() at the start of each request,
record() at each LLM invocation, and get_count()/get_calls() at the end.
"""
import threading

_state = threading.local()


def reset():
    _state.count = 0
    _state.calls = []


def record(caller: str, model: str, elapsed_ms: int):
    if not hasattr(_state, "count"):
        _state.count = 0
        _state.calls = []
    _state.count += 1
    entry = f"#{_state.count} {caller} ({model}) {elapsed_ms}ms"
    _state.calls.append(entry)
    print(f"[LLM_CALL #{_state.count}] caller={caller}  model={model}  took {elapsed_ms}ms")


def get_count() -> int:
    return getattr(_state, "count", 0)


def get_calls() -> list:
    return list(getattr(_state, "calls", []))
