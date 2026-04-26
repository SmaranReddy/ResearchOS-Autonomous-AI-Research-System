"""
Analyze benchmark results and produce:
  1. Comparison table (Baseline vs Adaptive cold vs Adaptive warm)
  2. Per-metric % improvements
  3. 3–4 resume bullet points built from REAL numbers only

Usage
-----
    python backend/evaluation/analyze_results.py
    python backend/evaluation/analyze_results.py --results path/to/benchmark_results.json
"""

import sys
import os
import json
import statistics
import argparse
from typing import List, Dict, Any, Optional

RESULTS_DEFAULT = os.path.join(
    os.path.dirname(__file__), "results", "benchmark_results.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(record: Dict, *keys) -> Any:
    node = record
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node


def _series(results: List[Dict], *path) -> List:
    out = []
    for r in results:
        v = _get(r, *path)
        if v is not None:
            out.append(v)
    return out


def _mean(lst) -> float:
    lst = [x for x in lst if x is not None]
    return round(statistics.mean(lst), 2) if lst else 0.0


def _median(lst) -> float:
    lst = [x for x in lst if x is not None]
    return round(statistics.median(lst), 2) if lst else 0.0


def _rate(lst, pred) -> float:
    if not lst:
        return 0.0
    return round(sum(1 for x in lst if pred(x)) / len(lst), 3)


def _pct(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return round((new - old) / old * 100, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict:
    # Latency
    b_lat  = _series(results, "baseline",      "latency", "total_ms")
    ac_lat = _series(results, "adaptive_cold", "latency", "total_ms")
    aw_lat = _series(results, "adaptive_warm", "latency", "total_ms")

    # LLM calls
    b_calls  = _series(results, "baseline",      "llm_calls")
    ac_calls = _series(results, "adaptive_cold", "llm_calls")
    aw_calls = _series(results, "adaptive_warm", "llm_calls")

    # Quality
    b_rel  = _series(results, "baseline",      "quality", "relevance")
    ac_rel = _series(results, "adaptive_cold", "quality", "relevance")
    b_cor  = _series(results, "baseline",      "quality", "correctness")
    ac_cor = _series(results, "adaptive_cold", "quality", "correctness")
    b_kw   = _series(results, "baseline",      "quality", "keyword_coverage")
    ac_kw  = _series(results, "adaptive_cold", "quality", "keyword_coverage")

    # Grounding
    b_gr_list  = _series(results, "baseline",      "quality", "grounded")
    ac_gr_list = _series(results, "adaptive_cold", "quality", "grounded")

    # Retry & cache
    ac_retry    = _series(results, "adaptive_cold", "retry_triggered")
    aw_hits     = _series(results, "adaptive_warm", "cache_hit")
    ac_conf     = _series(results, "adaptive_cold", "confidence_score")

    # Failure (non-success status)
    b_status  = _series(results, "baseline",      "status")
    ac_status = _series(results, "adaptive_cold", "status")

    return {
        "n": len(results),
        "latency": {
            "b_avg":  _mean(b_lat),   "b_median":  _median(b_lat),
            "ac_avg": _mean(ac_lat),  "ac_median": _median(ac_lat),
            "aw_avg": _mean(aw_lat),  "aw_median": _median(aw_lat),
        },
        "llm_calls": {
            "b_avg":  _mean(b_calls),
            "ac_avg": _mean(ac_calls),
            "aw_avg": _mean(aw_calls),
        },
        "quality": {
            "b_rel":  _mean(b_rel),   "ac_rel":  _mean(ac_rel),
            "b_cor":  _mean(b_cor),   "ac_cor":  _mean(ac_cor),
            "b_kw":   _mean(b_kw),    "ac_kw":   _mean(ac_kw),
        },
        "grounding": {
            "b_rate":  _rate(b_gr_list,  bool),
            "ac_rate": _rate(ac_gr_list, bool),
        },
        "retry_rate":     _rate(ac_retry, bool),
        "cache_hit_rate": _rate(aw_hits,  bool),
        "failure_rate": {
            "b":  _rate(b_status,  lambda s: s not in ("success",)),
            "ac": _rate(ac_status, lambda s: s not in ("success",)),
        },
        "confidence": {
            "ac_avg":    _mean(ac_conf),
            "ac_median": _median(ac_conf),
        },
        "retry_count":      sum(1 for x in ac_retry if x),
        "cache_hit_count":  sum(1 for x in aw_hits  if x),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _hr(ch="─", w=74):
    print(ch * w)


def _arrow(old, new, lower_is_better=False):
    """Return a coloured-text % change string."""
    d = _pct(old, new)
    if lower_is_better:
        sign = "▼" if d <= 0 else "▲"
    else:
        sign = "▲" if d >= 0 else "▼"
    return f"{sign} {abs(d):.1f}%"


def print_table(m: Dict) -> None:
    lat  = m["latency"]
    llm  = m["llm_calls"]
    qual = m["quality"]
    gr   = m["grounding"]
    fail = m["failure_rate"]

    _hr("═")
    print("  BENCHMARK RESULTS — Baseline RAG  vs  Adaptive RAG")
    print(f"  n = {m['n']} queries  (factual / vague / multi-hop / adversarial)")
    _hr("═")
    print(f"\n  {'Metric':<40} {'Baseline':>10} {'Adap.Cold':>12} {'Adap.Warm':>12} {'Δ (cold)':>10}")
    _hr()

    def row(label, b, ac, aw="-", low=False):
        d = _arrow(b, ac, lower_is_better=low) if isinstance(b, (int, float)) and isinstance(ac, (int, float)) else ""
        print(f"  {label:<40} {str(b):>10} {str(ac):>12} {str(aw):>12} {d:>10}")

    row("Avg total latency (ms)",         lat["b_avg"],  lat["ac_avg"],  lat["aw_avg"], low=True)
    row("Median total latency (ms)",       lat["b_median"],lat["ac_median"],lat["aw_median"], low=True)
    row("Avg LLM calls / query",           llm["b_avg"],  llm["ac_avg"],  llm["aw_avg"], low=True)
    row("Relevance score (avg /10)",       qual["b_rel"], qual["ac_rel"])
    row("Correctness score (avg /10)",     qual["b_cor"], qual["ac_cor"])
    row("Keyword coverage (avg)",         f"{qual['b_kw']:.2f}", f"{qual['ac_kw']:.2f}")
    row("Grounding rate",                 f"{gr['b_rate']:.0%}", f"{gr['ac_rate']:.0%}")
    row("Failure rate",                   f"{fail['b']:.0%}", f"{fail['ac']:.0%}", low=True)
    row("Cache hit rate",                 "N/A", "N/A", f"{m['cache_hit_rate']:.0%}")
    row("Retry rate (adaptive only)",     "0%", f"{m['retry_rate']:.0%}", "-")
    row("Avg confidence score",           "N/A", f"{m['confidence']['ac_avg']:.2f}", "-")

    _hr()
    print()


def print_improvements(m: Dict) -> None:
    lat  = m["latency"]
    qual = m["quality"]
    gr   = m["grounding"]
    fail = m["failure_rate"]
    llm  = m["llm_calls"]

    print("  KEY % IMPROVEMENTS  (Adaptive vs Baseline)\n")

    def _line(label, old, new, low=False, unit=""):
        d = _pct(old, new)
        arrow = ("▼" if d <= 0 else "▲") if low else ("▲" if d >= 0 else "▼")
        print(f"    {label:<38}  {old}{unit} → {new}{unit}   {arrow} {abs(d):.1f}%")

    _line("Answer relevance",       qual["b_rel"],  qual["ac_rel"],  unit="/10")
    _line("Answer correctness",     qual["b_cor"],  qual["ac_cor"],  unit="/10")
    _line("Keyword coverage",       f"{qual['b_kw']:.2f}", f"{qual['ac_kw']:.2f}")
    _line("Grounding rate",         f"{gr['b_rate']:.0%}", f"{gr['ac_rate']:.0%}")
    _line("Failure rate",           f"{fail['b']:.0%}", f"{fail['ac']:.0%}", low=True)
    print()
    _line("Latency — cached query vs baseline",
          f"{lat['b_avg']:.0f}", f"{lat['aw_avg']:.0f}", low=True, unit="ms")
    _line("LLM calls — cached vs baseline",
          f"{llm['b_avg']:.1f}", f"{llm['aw_avg']:.1f}", low=True)

    print(f"\n    Cache hit rate  : {m['cache_hit_rate']:.0%}  ({m['cache_hit_count']}/{m['n']} queries)")
    print(f"    Retry rate      : {m['retry_rate']:.0%}  ({m['retry_count']}/{m['n']} queries triggered retry)")
    _hr()
    print()


def print_resume_bullets(m: Dict) -> None:
    lat  = m["latency"]
    qual = m["quality"]
    gr   = m["grounding"]
    fail = m["failure_rate"]
    llm  = m["llm_calls"]

    rel_pct  = abs(_pct(qual["b_rel"], qual["ac_rel"]))
    cor_pct  = abs(_pct(qual["b_cor"], qual["ac_cor"]))
    kw_pct   = abs(_pct(qual["b_kw"],  qual["ac_kw"]))
    warm_pct = abs(_pct(lat["b_avg"],  lat["aw_avg"]))
    fail_pp  = round((fail["b"] - fail["ac"]) * 100, 0)
    gr_pp    = round((gr["ac_rate"] - gr["b_rate"]) * 100, 0)
    n        = m["n"]
    cache_rt = m["cache_hit_rate"]
    retry_rt = m["retry_rate"]
    conf_avg = m["confidence"]["ac_avg"]

    print("  RESUME BULLET POINTS  (based on real benchmark data)\n")

    bullets: list[tuple[str, str]] = []

    # Bullet 1 — Quality improvement (only if meaningful)
    if rel_pct >= 5 or cor_pct >= 5:
        bullets.append((
            "Answer quality improvement via query expansion + batched reranking",
            f"Improved LLM-evaluated answer relevance by {rel_pct:.0f}% "
            f"({qual['b_rel']}/10 → {qual['ac_rel']}/10) and correctness by {cor_pct:.0f}% "
            f"({qual['b_cor']}/10 → {qual['ac_cor']}/10) over a single-pass baseline, "
            f"measured across a {n}-query benchmark covering factual, vague, multi-hop, "
            f"and adversarial queries."
        ))

    # Bullet 2 — Cache speedup (always meaningful if hits > 0)
    if cache_rt > 0:
        bullets.append((
            "Latency reduction via LRU + TTL response caching",
            f"Reduced repeated-query latency by {warm_pct:.0f}% "
            f"({lat['b_avg']:.0f} ms → {lat['aw_avg']:.0f} ms) "
            f"via an LRU + 10-min TTL response cache with SHA-256 key and index-version "
            f"invalidation, achieving a {cache_rt:.0%} cache hit rate on warm runs "
            f"({m['cache_hit_count']}/{n} queries served from cache with 0 LLM calls)."
        ))

    # Bullet 3 — Failure rate / grounding (only if non-trivial improvement)
    if fail_pp > 0 or gr_pp > 0:
        line = (
            f"Reduced query failure rate from {fail['b']:.0%} to {fail['ac']:.0%} "
            f"(−{fail_pp:.0f} pp) and raised grounding rate from "
            f"{gr['b_rate']:.0%} to {gr['ac_rate']:.0%} (+{gr_pp:.0f} pp) using "
            f"HyDE-augmented parallel Pinecone retrieval (3× query fan-out) and O(1) "
            f"batched LLM reranking."
        )
        bullets.append(("Retrieval quality improvement reduces failures", line))

    # Bullet 4 — Retry (only if it actually triggered)
    if retry_rt > 0:
        bullets.append((
            "Self-correcting retry loop improves answer quality",
            f"Implemented a type-aware self-critique retry mechanism "
            f"(incomplete / incorrect / not_grounded strategies) that triggered on "
            f"{retry_rt:.0%} of queries ({m['retry_count']}/{n}), accepted only when "
            f"improvement exceeded a 0.05 threshold, with avg confidence {conf_avg:.2f} "
            f"on successful responses."
        ))

    # Fallback bullet if nothing else qualifies
    if not bullets:
        bullets.append((
            "Adaptive RAG with caching and multi-query retrieval",
            f"Built an adaptive RAG pipeline featuring HyDE query expansion, "
            f"parallel Pinecone retrieval (3× fan-out), O(1) batched reranking, "
            f"LRU+TTL response caching, and a self-critique retry loop; benchmarked "
            f"against a minimal baseline across {n} queries."
        ))

    for i, (title, text) in enumerate(bullets, 1):
        _hr()
        print(f"  [{i}]  {title}")
        # word-wrap at 80 chars
        words = text.split()
        line_ = "       "
        for w in words:
            if len(line_) + len(w) + 1 > 79:
                print(line_)
                line_ = "       " + w + " "
            else:
                line_ += w + " "
        if line_.strip():
            print(line_)

    _hr()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(results_path: str = RESULTS_DEFAULT) -> None:
    if not os.path.exists(results_path):
        print(f"[ERROR] Results file not found:\n  {results_path}")
        print("Run  python backend/evaluation/benchmark_runner.py  first.")
        sys.exit(1)

    with open(results_path) as fh:
        results = json.load(fh)

    if not results:
        print("[ERROR] Results file is empty.")
        sys.exit(1)

    m = compute_metrics(results)

    # Save summary
    metrics_path = os.path.join(os.path.dirname(results_path), "metrics_summary.json")
    with open(metrics_path, "w") as fh:
        json.dump(m, fh, indent=2)

    print()
    print_table(m)
    print_improvements(m)
    print_resume_bullets(m)
    print(f"  Full metrics → {metrics_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_DEFAULT)
    args = parser.parse_args()
    main(args.results)
