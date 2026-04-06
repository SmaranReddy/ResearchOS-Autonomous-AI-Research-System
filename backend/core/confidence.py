"""
Composite confidence scoring.

Combines three complementary signals already available after the rerank step:
  1. Retrieval cosine similarity  (Pinecone, 0–1)   — raw relevance
  2. Rerank quality score         (LLM 0–10 → 0–1)  — semantic precision
  3. LLM context sufficiency      (Groq, 0–1)        — answer-grounding

No additional LLM calls are made — the LLM score is reused from Phase 2c
(cached on state.confidence) or from generate_answer's internal call.

Adaptive weights
-----------------
Instead of fixed weights, _adaptive_weights() adjusts the mix based on signal
quality observed at inference time:
  - Very few docs     → retrieval signal unreliable → shift weight to rerank
  - High rerank spread → rerank discriminates well → increase its weight
  - LLM score < 0.30  → LLM unsure → reduce its weight, redistribute
"""

from __future__ import annotations
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.state import State

# Default (base) weights — sum must equal 1.0
_W_RETRIEVAL: float = 0.25
_W_RERANK:    float = 0.35
_W_LLM:       float = 0.40

# Score thresholds for status determination
LOW_CONFIDENCE_THRESHOLD: float = 0.40    # below → "low_confidence" status
ANSWER_CONFIDENCE_THRESHOLD: float = 0.50  # must match AnswerAgent.CONFIDENCE_THRESHOLD


def _adaptive_weights(
    n_docs: int,
    rerank_scores: list[float],
    llm_score: float,
) -> tuple[float, float, float]:
    """
    Return (w_retrieval, w_rerank, w_llm) adjusted to current signal quality.
    Weights are always normalised so they sum exactly to 1.0.

    Rules (applied in order, each adjusts a delta from the base):
      1. n_docs < 3  → retrieval has few data-points; shift 0.10 from retrieval → rerank
      2. rerank std > 3.0  → rerank discriminates well; add 0.05 to rerank
      3. llm_score < 0.30  → LLM uncertain; shift 0.10 from LLM → retrieval + rerank equally
    """
    w_ret = _W_RETRIEVAL
    w_rer = _W_RERANK
    w_llm = _W_LLM

    # Rule 1: too few docs → retrieval signal unreliable
    if n_docs < 3:
        w_ret -= 0.10
        w_rer += 0.10

    # Rule 2: high rerank spread → rerank is a reliable discriminator
    if len(rerank_scores) >= 2:
        try:
            std = statistics.stdev(rerank_scores)
        except statistics.StatisticsError:
            std = 0.0
        if std > 3.0:
            w_rer += 0.05
            w_llm -= 0.05

    # Rule 3: LLM is unsure → reduce its weight
    if llm_score < 0.30:
        shift = 0.10
        w_llm -= shift
        w_ret += shift / 2.0
        w_rer += shift / 2.0

    # Clamp to [0, 1] before normalisation
    w_ret = max(0.0, w_ret)
    w_rer = max(0.0, w_rer)
    w_llm = max(0.0, w_llm)

    total = w_ret + w_rer + w_llm
    if total == 0.0:
        return _W_RETRIEVAL, _W_RERANK, _W_LLM  # fallback to defaults

    return w_ret / total, w_rer / total, w_llm / total


def _describe_retrieval_quality(
    n_docs: int,
    retrieval_norm: float,
    rerank_scores: list[float],
    rerank_norm: float,
) -> str:
    """
    Produce a one-sentence summary of retrieval quality using only
    the signals already computed — no extra LLM calls.
    """
    variance = 0.0
    if len(rerank_scores) >= 2:
        mean = sum(rerank_scores) / len(rerank_scores)
        variance = sum((s - mean) ** 2 for s in rerank_scores) / len(rerank_scores)

    if n_docs < 3:
        return (
            f"Limited document count ({n_docs} docs) reduces confidence in retrieval quality."
        )
    if retrieval_norm >= 0.7 and rerank_norm >= 0.6 and variance < 4.0:
        return (
            f"High semantic overlap (retrieval={retrieval_norm:.2f}) with strong, "
            f"consistent rerank scores (mean={rerank_norm*10:.1f}/10, variance={variance:.1f})."
        )
    if variance > 9.0:
        return (
            f"Low semantic overlap with high variance in rerank scores "
            f"(retrieval={retrieval_norm:.2f}, variance={variance:.1f}), "
            f"indicating mixed document relevance."
        )
    return (
        f"Moderate retrieval quality: {n_docs} docs, "
        f"retrieval={retrieval_norm:.2f}, rerank mean={rerank_norm*10:.1f}/10."
    )


def _describe_confidence_reasoning(
    retrieval_norm: float,
    rerank_norm: float,
    llm_score: float,
    composite: float,
) -> str:
    """
    Produce a one-sentence explanation of the final composite confidence.
    Uses only signals already on hand — no extra LLM calls.
    """
    signals = {
        "retrieval": retrieval_norm,
        "rerank":    rerank_norm,
        "llm":       llm_score,
    }
    weakest  = min(signals, key=signals.__getitem__)
    strongest = max(signals, key=signals.__getitem__)

    if composite >= 0.70:
        return (
            f"High confidence: strong {strongest} signal ({signals[strongest]:.2f}) "
            f"supported by consistent retrieval and rerank quality."
        )
    if composite >= 0.40:
        return (
            f"Moderate confidence: {weakest} signal ({signals[weakest]:.2f}) is the "
            f"limiting factor; other signals are adequate."
        )
    return (
        f"Low confidence: weak {weakest} signal ({signals[weakest]:.2f}) — "
        f"retrieved documents may not cover the query well."
    )


def compute_composite(state: "State") -> float:
    """
    Compute composite confidence from the three signals and write it back
    to state.confidence.  Returns the composite score.

    Call AFTER generate_answer (which sets state.confidence = llm_score).
    """
    # Signal 1: mean retrieval cosine similarity (Pinecone, 0–1)
    retrieval_scores = [
        d.get("score", 0.0)
        for d in state.ranked_docs
        if isinstance(d, dict) and d.get("score") is not None
    ]
    retrieval_norm = (
        sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    )

    # Signal 2: top-3 rerank scores (0–10 → 0–1)
    rerank_raw = sorted(
        [d.get("rerank_score", 0.0) for d in state.ranked_docs if isinstance(d, dict)],
        reverse=True,
    )
    top_rerank = rerank_raw[:3]
    rerank_norm = (
        sum(top_rerank) / (10.0 * len(top_rerank)) if top_rerank else 0.0
    )

    # Signal 3: LLM-rated context sufficiency (reuse, no extra call)
    llm_score = max(0.0, min(1.0, state.confidence))

    # Adaptive weight selection
    all_rerank = [d.get("rerank_score", 0.0) for d in state.ranked_docs if isinstance(d, dict)]
    w_ret, w_rer, w_llm = _adaptive_weights(len(retrieval_scores), all_rerank, llm_score)

    composite = (
        w_ret * retrieval_norm
        + w_rer * rerank_norm
        + w_llm * llm_score
    )
    composite = round(max(0.0, min(1.0, composite)), 4)

    print(
        f"[CONFIDENCE] retrieval={retrieval_norm:.3f}×{w_ret:.2f}"
        f"  rerank={rerank_norm:.3f}×{w_rer:.2f}"
        f"  llm={llm_score:.3f}×{w_llm:.2f}"
        f"  → composite={composite:.3f}"
    )

    state.confidence = composite

    # --- Decision trace: retrieval quality + confidence reasoning ---
    state.decision_trace["retrieval_quality"] = _describe_retrieval_quality(
        len(retrieval_scores), retrieval_norm, all_rerank, rerank_norm
    )
    state.decision_trace["confidence_reasoning"] = _describe_confidence_reasoning(
        retrieval_norm, rerank_norm, llm_score, composite
    )

    return composite


def derive_status(state: "State") -> str:
    """
    Map pipeline outcome to a human-readable status string.
    Called after compute_composite so state.confidence is the composite score.
    """
    if state.is_fallback:
        return "fallback"
    if state.confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return "success"
