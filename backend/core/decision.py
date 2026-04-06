"""
Adaptive ingestion decision policy.

Replaces three separate static threshold checks with a single, interpretable
multi-signal function.  No ML model — pure rule-based logic using signals
that are already computed in the pipeline.

Rules (evaluated in order, first match wins):
  0. Entity coverage   — key query entities missing from top-doc text
  1. Low coverage      — n_docs < 3
  2. Weak retrieval    — retrieval_norm < 0.3 AND rerank_norm < 0.4
  3. High rerank noise — rerank variance > 9.0 AND rerank_norm < 0.5
  4. LLM uncertainty   — llm_score < 0.3 AND rerank_norm < 0.6
  B. Borderline        — signals in ambiguous range (no ingestion triggered)

Signals that are not yet available at a given pipeline phase are passed as
None — the corresponding rules are automatically skipped.

Return value
------------
(should_ingest: bool, reason: str, strength: str)
  reason   ∈ {"low_docs", "low_relevance", "low_confidence",
               "borderline", "sufficient_context"}
  strength ∈ {"strong", "moderate", "weak"}
             "strong"   → signal clearly exceeds threshold
             "moderate" → signal exceeds threshold but is not far from it
             "weak"     → borderline / ambiguous case
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Variance threshold for Rule 3 (std > 3.0  ↔  variance > 9.0)
_VARIANCE_THRESHOLD: float = 9.0

# Common words that are NOT meaningful entities for coverage checking
_STOP_WORDS = {
    "what", "how", "why", "when", "where", "which", "who", "are", "is", "the",
    "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "from",
    "do", "does", "did", "can", "could", "would", "should", "will", "be",
    "between", "compare", "comparison", "difference", "different", "vs",
    "versus", "explain", "describe", "tell", "me", "about", "each", "other",
    "than", "that", "this", "these", "those", "them", "they", "their",
}

# Minimum entity length to be considered meaningful
_MIN_ENTITY_LEN = 3


def _extract_entities(query: str) -> List[str]:
    """
    Extract meaningful tokens from the query (lowercased, stop-words removed).
    Used for Rule 0 entity coverage check.
    """
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#]*\b", query.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) >= _MIN_ENTITY_LEN]


def _entity_in_text(entity: str, combined: str) -> bool:
    """
    Fuzzy word-boundary match to handle morphological variants.
    Checks exact substring first, then a prefix stem (handles plurals,
    -ing, -ed, -tion suffixes, e.g. "architectures" ↔ "architecture").
    """
    if entity in combined:
        return True
    # Stem check: entity prefix of length max(4, len-2) must appear as word-start
    stem_len = max(4, len(entity) - 2)
    if len(entity) >= 5:
        stem = entity[:stem_len]
        # Check if stem appears at a word boundary in combined
        import re as _re
        if _re.search(r"\b" + _re.escape(stem), combined):
            return True
    return False


def _docs_cover_entities(entities: List[str], docs: List[dict]) -> tuple[bool, List[str]]:
    """
    Check whether the combined text of retrieved docs mentions every key entity.
    Uses fuzzy matching to handle plurals and morphological variants.
    Returns (all_covered: bool, missing_entities: List[str]).
    """
    if not entities or not docs:
        return True, []

    combined = " ".join(
        (d.get("text", "") or d.get("metadata", {}).get("text", "")).lower()
        for d in docs
    )
    missing = [e for e in entities if not _entity_in_text(e, combined)]
    return len(missing) == 0, missing


def should_trigger_ingestion(
    n_docs: int,
    retrieval_norm: float,
    rerank_scores: Optional[List[float]] = None,
    rerank_norm: Optional[float] = None,
    llm_score: Optional[float] = None,
    query: Optional[str] = None,
    docs: Optional[List[dict]] = None,
) -> Tuple[bool, str, str]:
    """
    Decide whether ingestion should be triggered based on available signals.

    Parameters
    ----------
    n_docs          : number of retrieved documents
    retrieval_norm  : mean Pinecone cosine similarity (0–1)
    rerank_scores   : raw rerank scores (0–10); None if rerank hasn't run yet
    rerank_norm     : mean rerank score normalised to 0–1; None if not available
    llm_score       : LLM context-sufficiency score (0–1); None if not available
    query           : original (resolved) user query for entity coverage check
    docs            : retrieved documents to check entity coverage against

    Returns
    -------
    (should_ingest: bool, reason: str, strength: str)
    """

    # ------------------------------------------------------------------
    # Rule 0 — Entity coverage: key query entities absent from retrieved docs.
    # Catches cases where scores look acceptable but docs are topically mismatched.
    # Only runs when rerank has completed (docs quality is known).
    # ------------------------------------------------------------------
    if query and docs and rerank_norm is not None:
        entities = _extract_entities(query)
        covered, missing = _docs_cover_entities(entities, docs)
        if not covered:
            print(
                f"[DECISION] Rule 0 fired: missing entities {missing} in retrieved docs "
                f"→ low_relevance (strong)"
            )
            return True, "low_relevance", "strong"

    # ------------------------------------------------------------------
    # Rule 1 — Low coverage: too few docs to form a reliable answer
    # ------------------------------------------------------------------
    if n_docs < 3:
        strength = "strong" if n_docs == 0 else "moderate"
        print(f"[DECISION] Rule 1 fired: n_docs={n_docs} < 3 → low_docs ({strength})")
        return True, "low_docs", strength

    # ------------------------------------------------------------------
    # Rule 2 — Weak retrieval: low cosine similarity AND poor rerank quality
    # ------------------------------------------------------------------
    if rerank_norm is not None and retrieval_norm < 0.3 and rerank_norm < 0.4:
        strength = "strong" if retrieval_norm < 0.2 else "moderate"
        print(
            f"[DECISION] Rule 2 fired: retrieval_norm={retrieval_norm:.3f} < 0.3 "
            f"and rerank_norm={rerank_norm:.3f} < 0.4 → low_relevance ({strength})"
        )
        return True, "low_relevance", strength

    # ------------------------------------------------------------------
    # Rule 3 — High rerank noise: inconsistent scores signal mixed relevance
    # ------------------------------------------------------------------
    if rerank_scores is not None and rerank_norm is not None and len(rerank_scores) >= 2:
        mean = sum(rerank_scores) / len(rerank_scores)
        variance = sum((s - mean) ** 2 for s in rerank_scores) / len(rerank_scores)
        if variance > _VARIANCE_THRESHOLD and rerank_norm < 0.5:
            print(
                f"[DECISION] Rule 3 fired: rerank variance={variance:.2f} > {_VARIANCE_THRESHOLD} "
                f"and rerank_norm={rerank_norm:.3f} < 0.5 → low_relevance (moderate)"
            )
            return True, "low_relevance", "moderate"

    # ------------------------------------------------------------------
    # Rule 4 — LLM uncertainty: model unsure even after rerank passes
    # ------------------------------------------------------------------
    if llm_score is not None and rerank_norm is not None:
        if llm_score < 0.3 and rerank_norm < 0.6:
            strength = "strong" if llm_score < 0.2 else "moderate"
            print(
                f"[DECISION] Rule 4 fired: llm_score={llm_score:.3f} < 0.3 "
                f"and rerank_norm={rerank_norm:.3f} < 0.6 → low_confidence ({strength})"
            )
            return True, "low_confidence", strength

    # ------------------------------------------------------------------
    # Borderline — signals are not alarming but are not clearly sufficient.
    # No ingestion triggered; strength is always "weak".
    # Borderline is a weak signal — must not override stronger failure conditions above.
    # Placed AFTER rules 1–4 so stronger rules always win.
    # ------------------------------------------------------------------
    if rerank_norm is not None and (0.3 <= retrieval_norm <= 0.4) and (0.4 <= rerank_norm <= 0.5):
        print(
            f"[DECISION] Borderline: retrieval_norm={retrieval_norm:.3f} ∈ [0.3, 0.4] "
            f"and rerank_norm={rerank_norm:.3f} ∈ [0.4, 0.5] → borderline (weak)"
        )
        return False, "borderline", "weak"

    # ------------------------------------------------------------------
    # Sufficient context — estimate strength from distance to thresholds
    # ------------------------------------------------------------------
    near_threshold = (
        retrieval_norm < 0.45
        or (rerank_norm is not None and rerank_norm < 0.55)
        or (llm_score is not None and llm_score < 0.45)
    )
    strength = "moderate" if near_threshold else "strong"
    print(
        f"[DECISION] all rules passed: n_docs={n_docs}, "
        f"retrieval={retrieval_norm:.3f}, "
        f"rerank_norm={rerank_norm if rerank_norm is not None else 'n/a'}, "
        f"llm={llm_score if llm_score is not None else 'n/a'} → sufficient_context ({strength})"
    )
    return False, "sufficient_context", strength


# Reason → action-string mapping (matches decision_trace["action"] values)
REASON_TO_ACTION: dict[str, str] = {
    "low_docs":              "triggered_ingestion_low_docs",
    "low_relevance":         "triggered_ingestion_low_relevance",
    "low_confidence":        "triggered_ingestion_low_confidence",
    "borderline":            "used_existing_knowledge_borderline",
    "sufficient_context":    "used_existing_knowledge",
}
