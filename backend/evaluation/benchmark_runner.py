"""
Benchmark runner: Baseline RAG vs Adaptive RAG

Measures latency, LLM call count, quality, caching, and retry across
30 queries (factual / vague / multi-hop / adversarial).

KEY DESIGN DECISIONS
--------------------
* Cache is consulted/written explicitly here (same logic as app.py /query),
  because run_pipeline() bypasses the cache layer.
* Warm run = cache.get() ONLY — no pipeline re-execution on a hit.
  Zero LLM calls on a cache hit is the expected result.
* LLM calls are counted via a class-level monkey-patch applied before
  any Groq client is created.
* INTER_QUERY_DELAY_S respects Groq's 30 RPM free-tier limit.

Usage
-----
    cd ResearchOS-Autonomous-AI-Research-System
    python backend/evaluation/benchmark_runner.py

Output
------
    backend/evaluation/results/benchmark_results.json
    backend/evaluation/results/benchmark_results.csv
"""

import sys
import os
import json
import csv
import time
import threading
from typing import Dict, Any

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".env")
))

# ── LLM call counter — MUST be patched before Groq clients are instantiated ──
_call_count = 0
_call_lock  = threading.Lock()


def _reset_call_count() -> None:
    global _call_count
    with _call_lock:
        _call_count = 0


def _get_call_count() -> int:
    with _call_lock:
        return _call_count


_llm_counter_installed = False
try:
    from groq.resources.chat.completions import Completions as _GCompl
    _orig_create = _GCompl.create

    def _counted_create(self, *args, **kwargs):
        global _call_count
        with _call_lock:
            _call_count += 1
        return _orig_create(self, *args, **kwargs)

    _GCompl.create = _counted_create
    _llm_counter_installed = True
    print("[PATCH] Groq LLM call counter installed.")
except Exception as _e:
    print(f"[WARN] Could not install LLM call counter: {_e}")

# ── Project imports (after patch so all Groq clients use patched class) ───────
from evaluation.baseline_rag import BaselineRAG
from evaluation.llm_evaluator import LLMEvaluator
from core.executor import run_pipeline
from core.cache import get_cache

# ── Constants ─────────────────────────────────────────────────────────────────
QUERIES_PATH = os.path.join(os.path.dirname(__file__), "benchmark_queries.json")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "benchmark_results.json")
RESULTS_CSV  = os.path.join(RESULTS_DIR, "benchmark_results.csv")

# Delay between LLM-calling steps to stay under Groq 30 RPM free-tier.
# Baseline run: 1 LLM call  (~2 RPM used)
# LLM quality eval: 1 LLM call (~2 RPM used)
# Adaptive cold run: 4-7 LLM calls (~10 RPM used)
# Total per query: ~5-10 calls → 2.5s delay keeps us safely under limit.
INTER_QUERY_DELAY_S = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Quality helpers (rule-based, no extra LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_coverage(answer: str, keywords) -> float:
    if not keywords:
        return 1.0
    al = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in al)
    return round(hits / len(keywords), 3)


def _is_grounded(answer: str, keywords) -> bool:
    """At least 25 % of expected keywords appear → grounded."""
    return _keyword_coverage(answer, keywords) >= 0.25


def _is_failure(status: str) -> bool:
    return status in ("low_confidence", "fallback", "error")


# ─────────────────────────────────────────────────────────────────────────────
# Baseline runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_baseline(baseline: BaselineRAG, query: str) -> Dict:
    _reset_call_count()
    raw = baseline.run(query)
    raw["llm_calls"] = _get_call_count()   # actual count, not hardcoded 1
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive runners (with explicit cache integration)
# ─────────────────────────────────────────────────────────────────────────────

def _run_adaptive_cold(query: str) -> Dict:
    """
    Full adaptive pipeline — cache already cleared at benchmark start.
    On success, writes the result dict to the cache so the warm run hits it.
    """
    cache   = get_cache()
    history = []

    _reset_call_count()
    t_start = time.monotonic()

    answer, confidence, _, sources, latency_ms, status, decision_trace, retried = run_pipeline(
        query, num_papers=3, chat_history=None, disable_retry=False
    )
    wall_ms = int((time.monotonic() - t_start) * 1000)

    result = {
        "answer":           answer,
        "confidence_score": round(confidence, 3),
        "docs_retrieved":   len(sources),
        "status":           status,
        "latency": {
            "transform_ms": latency_ms.get("transform_ms", 0),
            "retrieve_ms":  latency_ms.get("retrieve_ms", 0),
            "rerank_ms":    latency_ms.get("rerank_ms", 0),
            "llm_ms":       latency_ms.get("llm_ms", 0),
            "total_ms":     latency_ms.get("total_ms", wall_ms),
        },
        "llm_calls":       _get_call_count(),
        "retry_triggered": retried,
        "cache_hit":       False,
        "decision_trace":  decision_trace,
    }

    # Mirror app.py cache write: only cache "success" responses
    if status == "success":
        cache.set(query, history, result)

    return result


def _run_adaptive_warm(query: str, cold_result: Dict) -> Dict:
    """
    Warm run — reads directly from the cache.
    If the cold run was NOT 'success' (cache miss), reuse the cold result
    and mark cache_hit=False so the stats are honest.
    """
    cache   = get_cache()
    history = []

    _reset_call_count()
    t_start = time.monotonic()
    cached  = cache.get(query, history)
    lookup_ms = max(1, int((time.monotonic() - t_start) * 1000))

    if cached is not None:
        return {
            "answer":           cached.get("answer", ""),
            "confidence_score": cached.get("confidence_score", 0.0),
            "docs_retrieved":   cached.get("docs_retrieved", 0),
            "status":           cached.get("status", "success"),
            "latency":          {"total_ms": lookup_ms},
            "llm_calls":        0,
            "retry_triggered":  False,
            "cache_hit":        True,
        }

    # Non-success cold run → cache was never written → genuine cache miss
    return {
        "answer":           cold_result["answer"],
        "confidence_score": cold_result["confidence_score"],
        "docs_retrieved":   cold_result["docs_retrieved"],
        "status":           cold_result["status"],
        "latency":          {"total_ms": cold_result["latency"]["total_ms"]},
        "llm_calls":        cold_result["llm_calls"],
        "retry_triggered":  cold_result["retry_triggered"],
        "cache_hit":        False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quality evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_quality(
    evaluator: LLMEvaluator,
    query: str,
    answer: str,
    keywords,
) -> Dict:
    scores = evaluator.evaluate(query, answer)
    return {
        "relevance":          scores.get("relevance", 0),
        "relevance_reason":   scores.get("relevance_reason", ""),
        "correctness":        scores.get("correctness", 0),
        "correctness_reason": scores.get("correctness_reason", ""),
        "grounded":           _is_grounded(answer, keywords),
        "keyword_coverage":   _keyword_coverage(answer, keywords),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(queries_path: str = QUERIES_PATH) -> list:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Wipe the response cache so every cold run is a genuine miss
    cache = get_cache()
    cache._store.clear()
    print("[BENCHMARK] Response cache cleared — cold runs will be cache-miss.\n")

    with open(queries_path) as fh:
        test_cases = json.load(fh)

    baseline  = BaselineRAG()
    evaluator = LLMEvaluator()
    results   = []
    total     = len(test_cases)

    print(f"{'='*70}")
    print(f"  BENCHMARK: {total} queries × (baseline | adaptive cold | adaptive warm)")
    print(f"{'='*70}\n")

    for idx, case in enumerate(test_cases, 1):
        qid      = case["id"]
        query    = case["query"]
        qtype    = case["type"]
        keywords = case.get("expected_keywords", [])

        print(f"\n[{idx:02d}/{total}] ({qtype.upper()})  {query[:80]}")
        print("─" * 70)

        # ── 1. Baseline ───────────────────────────────────────────────────────
        print("  [BASELINE]       running pipeline …")
        b_raw = _run_baseline(baseline, query)
        time.sleep(INTER_QUERY_DELAY_S)

        print("  [BASELINE]       evaluating quality …")
        b_quality = _evaluate_quality(evaluator, query, b_raw["answer"], keywords)
        time.sleep(INTER_QUERY_DELAY_S)

        print(
            f"  [BASELINE]       "
            f"total={b_raw['latency']['total_ms']}ms  "
            f"llm={b_raw['llm_calls']}  "
            f"rel={b_quality['relevance']}/10  "
            f"cor={b_quality['correctness']}/10  "
            f"status={b_raw['status']}"
        )

        # ── 2. Adaptive cold ─────────────────────────────────────────────────
        print("  [ADAPTIVE COLD]  running pipeline …")
        ac_raw = _run_adaptive_cold(query)
        time.sleep(INTER_QUERY_DELAY_S)

        print("  [ADAPTIVE COLD]  evaluating quality …")
        ac_quality = _evaluate_quality(evaluator, query, ac_raw["answer"], keywords)
        time.sleep(INTER_QUERY_DELAY_S)

        print(
            f"  [ADAPTIVE COLD]  "
            f"total={ac_raw['latency']['total_ms']}ms  "
            f"llm={ac_raw['llm_calls']}  "
            f"retry={ac_raw['retry_triggered']}  "
            f"conf={ac_raw['confidence_score']}  "
            f"rel={ac_quality['relevance']}/10  "
            f"cor={ac_quality['correctness']}/10  "
            f"status={ac_raw['status']}"
        )

        # ── 3. Adaptive warm (cache probe) ───────────────────────────────────
        print("  [ADAPTIVE WARM]  probing cache …")
        aw_raw = _run_adaptive_warm(query, ac_raw)

        print(
            f"  [ADAPTIVE WARM]  "
            f"total={aw_raw['latency']['total_ms']}ms  "
            f"cache_hit={aw_raw['cache_hit']}  "
            f"llm={aw_raw['llm_calls']}"
        )

        # ── Assemble record ───────────────────────────────────────────────────
        record = {
            "id":    qid,
            "query": query,
            "type":  qtype,
            "baseline":       {**b_raw,  "quality": b_quality},
            "adaptive_cold":  {**ac_raw, "quality": ac_quality},
            "adaptive_warm":  aw_raw,
        }
        results.append(record)

        # Incremental save — resume-safe if run interrupted
        _save_json(results, RESULTS_JSON)

        # Breathe before next query to stay under rate limit
        time.sleep(1.0)

    _save_csv(results, RESULTS_CSV)
    print(f"\n{'='*70}")
    print(f"  DONE — {total} queries completed.")
    print(f"  Results → {RESULTS_JSON}")
    print(f"  CSV     → {RESULTS_CSV}")
    print(f"{'='*70}\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(results: list, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)


def _save_csv(results: list, path: str) -> None:
    fieldnames = [
        "id", "query", "type",
        # baseline
        "b_total_ms", "b_retrieve_ms", "b_llm_ms",
        "b_llm_calls", "b_status",
        "b_relevance", "b_correctness", "b_grounded", "b_kw_coverage",
        # adaptive cold
        "ac_total_ms", "ac_transform_ms", "ac_retrieve_ms", "ac_rerank_ms", "ac_llm_ms",
        "ac_llm_calls", "ac_retry", "ac_confidence", "ac_status",
        "ac_relevance", "ac_correctness", "ac_grounded", "ac_kw_coverage",
        # adaptive warm
        "aw_total_ms", "aw_cache_hit", "aw_llm_calls",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            b  = r["baseline"]
            ac = r["adaptive_cold"]
            aw = r["adaptive_warm"]
            writer.writerow({
                "id":    r["id"],
                "query": r["query"],
                "type":  r["type"],
                # baseline
                "b_total_ms":    b["latency"]["total_ms"],
                "b_retrieve_ms": b["latency"]["retrieve_ms"],
                "b_llm_ms":      b["latency"]["llm_ms"],
                "b_llm_calls":   b["llm_calls"],
                "b_status":      b["status"],
                "b_relevance":   b["quality"]["relevance"],
                "b_correctness": b["quality"]["correctness"],
                "b_grounded":    b["quality"]["grounded"],
                "b_kw_coverage": b["quality"]["keyword_coverage"],
                # adaptive cold
                "ac_total_ms":    ac["latency"]["total_ms"],
                "ac_transform_ms":ac["latency"].get("transform_ms", 0),
                "ac_retrieve_ms": ac["latency"]["retrieve_ms"],
                "ac_rerank_ms":   ac["latency"]["rerank_ms"],
                "ac_llm_ms":      ac["latency"]["llm_ms"],
                "ac_llm_calls":   ac["llm_calls"],
                "ac_retry":       ac["retry_triggered"],
                "ac_confidence":  ac["confidence_score"],
                "ac_status":      ac["status"],
                "ac_relevance":   ac["quality"]["relevance"],
                "ac_correctness": ac["quality"]["correctness"],
                "ac_grounded":    ac["quality"]["grounded"],
                "ac_kw_coverage": ac["quality"]["keyword_coverage"],
                # adaptive warm
                "aw_total_ms":  aw["latency"]["total_ms"],
                "aw_cache_hit": aw["cache_hit"],
                "aw_llm_calls": aw["llm_calls"],
            })


if __name__ == "__main__":
    run_benchmark()
