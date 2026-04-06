import time
from typing import List, Dict

from core.state import State
from core.confidence import compute_composite, derive_status
from core.decision import should_trigger_ingestion, REASON_TO_ACTION
from core.critique import critique_answer

from utils.search import TavilyAgent
from ingestion.downloader import Downloader
from ingestion.preprocessing import Preprocessor
from ingestion.chunking import Chunker
from ingestion.embeddings import Embedder
from ingestion.indexing import Indexer
from retrieval.query_transform import QueryTransformer
from retrieval.retriever import RetrieverAgent
from retrieval.reranker import Reranker
from agents.answer_agent import AnswerAgent
from agents.critique_agent import CritiqueAgent


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _step_search_web(state: State) -> State:
    print(f"[search_web] query='{state.user_query}', limit={state.num_papers}")
    state.papers = TavilyAgent().search(state.user_query, max_results=state.num_papers)
    return state


def _step_download(state: State) -> State:
    print(f"[download] downloading {len(state.papers)} papers")
    downloader = Downloader()
    state.papers = [downloader.download_and_extract(p) for p in state.papers]
    return state


def _step_preprocess(state: State) -> State:
    print("[preprocess] cleaning extracted text")
    preprocessor = Preprocessor()
    for p in state.papers:
        if p.get("clean_text"):
            print(f"  [preprocess] skip '{p.get('title', 'unknown')}' — already preprocessed")
            continue
        print(f"  [preprocess] processing '{p.get('title', 'unknown')}'")
        p["clean_text"] = preprocessor.preprocess(p.get("full_text", ""))
    return state


def _step_chunk(state: State) -> State:
    print("[chunk] splitting text into chunks")
    chunker = Chunker(chunk_size=1000, overlap=150)
    for p in state.papers:
        if p.get("chunks"):
            print(f"  [chunk] skip '{p.get('title', 'unknown')}' — already chunked ({len(p['chunks'])} chunks)")
            continue
        print(f"  [chunk] chunking '{p.get('title', 'unknown')}'")
        p["chunks"] = chunker.chunk_text(p.get("clean_text", ""))
    return state


def _step_embed(state: State) -> State:
    print("[embed] embedding chunks")
    embedder = Embedder()
    for p in state.papers:
        if p.get("embeddings"):
            print(f"  [embed] skip '{p.get('title', 'unknown')}' — already embedded ({len(p['embeddings'])} vectors)")
            continue
        print(f"  [embed] embedding '{p.get('title', 'unknown')}'")
        p["embeddings"] = embedder.embed_chunks(p.get("chunks", []))
    return state


def _step_index(state: State) -> State:
    from core.cache import increment_index_version
    print("[index] uploading to Pinecone")
    indexer = Indexer()
    newly_indexed = 0
    for p in state.papers:
        if p.get("indexed"):
            print(f"  [index] skip '{p.get('title', 'unknown')}' — already indexed")
            continue
        print(f"  [index] indexing '{p.get('title', 'unknown')}'")
        indexer.index_chunks(
            p["title"],
            p.get("chunks", []),
            p.get("embeddings", []),
            url=p.get("link", ""),   # store source URL in Pinecone metadata
        )
        p["indexed"] = True
        newly_indexed += 1
    if newly_indexed > 0:
        increment_index_version()
    return state


def _step_query_transform(state: State) -> State:
    print("[query_transform] generating multi-query variations")
    state.rewritten_queries = QueryTransformer().transform(
        state.user_query,
        chat_history=state.chat_history if state.chat_history else None,
    )
    return state


def _step_retrieve(state: State) -> State:
    queries = state.rewritten_queries if state.rewritten_queries else [state.user_query]
    print(f"[retrieve] querying Pinecone with {len(queries)} queries")
    retriever = RetrieverAgent()
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.retrieve(q))
    print(f"[RETRIEVE] total docs before dedup: {len(all_docs)}")

    seen_ids = set()
    merged = []
    for doc in all_docs:
        key = doc.get("id") or doc.get("text", "")[:100]
        if key not in seen_ids:
            seen_ids.add(key)
            merged.append(doc)
    print(f"[RETRIEVE] after dedup: {len(merged)}")

    state.raw_docs = merged
    return state


def _step_rerank(state: State) -> State:
    print(f"[rerank] scoring {len(state.raw_docs)} docs")
    state.ranked_docs = Reranker().rerank(state.user_query, state.raw_docs)
    return state


def _step_answer(state: State) -> State:
    print("[answer] generating final answer")
    state.final_answer = AnswerAgent().generate_answer(
        state.user_query, state.ranked_docs, chat_history=state.chat_history, state=state
    )
    return state


def _step_critique(state: State) -> State:
    if state.is_fallback:
        print("[CRITIQUE] skipped — fallback response")
    else:
        print("[CRITIQUE] refining answer")
        context = [doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)]
        state.final_answer = CritiqueAgent().critique(state.final_answer, context, query=state.user_query)
    # append to history and keep only last 3
    state.chat_history.append({"query": state.user_query, "answer": state.final_answer})
    state.chat_history = state.chat_history[-3:]
    return state


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEP_REGISTRY = {
    "search_web":      _step_search_web,
    "download":        _step_download,
    "preprocess":      _step_preprocess,
    "chunk":           _step_chunk,
    "embed":           _step_embed,
    "index":           _step_index,
    "query_transform": _step_query_transform,
    "retrieve":        _step_retrieve,
    "rerank":          _step_rerank,
    "answer":          _step_answer,
    "critique":        _step_critique,
}

# Steps whose wall-clock time is tracked in state.latency_ms
_TIMED_STEPS = {
    "retrieve": "retrieve_ms",
    "rerank":   "rerank_ms",
    "answer":   "llm_ms",
    "critique": "llm_ms",
}


# ---------------------------------------------------------------------------
# Retrieval quality checks
# ---------------------------------------------------------------------------

MIN_DOCS = 3
MIN_AVG_SCORE = 0.3
MIN_RERANK_SCORE = 4.0

INGESTION_STEPS = ["search_web", "download", "preprocess", "chunk", "embed", "index"]


def _is_retrieval_weak(docs: list) -> bool:
    if len(docs) < MIN_DOCS:
        return True
    scores = [d.get("score") for d in docs if d.get("score") is not None]
    if scores and (sum(scores) / len(scores)) < MIN_AVG_SCORE:
        return True
    return False


def _is_relevance_low(ranked_docs: list) -> bool:
    scores = [d.get("rerank_score", 0.0) for d in ranked_docs if isinstance(d, dict)]
    if not scores:
        print("[INGESTION CHECK] no rerank scores available — treating as low relevance")
        return True

    sorted_scores = sorted(scores, reverse=True)
    total_docs = len(sorted_scores)
    median_score = sorted_scores[total_docs // 2]
    low_quality_count = sum(1 for s in scores if s < MIN_RERANK_SCORE)

    print(f"[INGESTION CHECK] scores: {[round(s, 1) for s in sorted_scores]}")
    print(f"[INGESTION CHECK] median: {median_score:.1f}")
    print(f"[INGESTION CHECK] low-quality docs (<{MIN_RERANK_SCORE}): {low_quality_count}/{total_docs}")

    relevance_low = median_score < MIN_RERANK_SCORE or low_quality_count >= total_docs // 2
    print(f"[INGESTION CHECK] decision: {'LOW' if relevance_low else 'SUFFICIENT'}")
    return relevance_low


def _run_step(state: State, step: str) -> State:
    print(f"\n[STEP] {step}")
    t0 = time.monotonic()
    state = STEP_REGISTRY[step](state)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Accumulate timing for observable stages
    key = _TIMED_STEPS.get(step)
    if key:
        state.latency_ms[key] = state.latency_ms.get(key, 0) + elapsed_ms

    if step == "search_web":
        print(f"[STEP] search_web → {len(state.papers)} papers found")
    elif step == "chunk":
        total = sum(len(p.get("chunks", [])) for p in state.papers)
        print(f"[STEP] chunk → {total} total chunks")
    elif step == "retrieve":
        print(f"[STEP] retrieve → {len(state.raw_docs)} docs retrieved ({elapsed_ms}ms)")
    elif step == "rerank":
        print(f"[STEP] rerank → top {len(state.ranked_docs)} docs ({elapsed_ms}ms)")
    elif step in ("answer", "critique"):
        print(f"[STEP] {step} done ({elapsed_ms}ms)")
    return state


# ---------------------------------------------------------------------------
# Self-critique retry helper
# ---------------------------------------------------------------------------

def _retry_with_expanded_context(state: State) -> State:
    """
    Type-aware retry: retrieval/rerank strategy depends on state._critique_type.

    incomplete    → more documents (top_k + 2) for broader coverage
    incorrect     → fewer, higher-precision documents (top_k reduced to 3)
    not_grounded  → standard retrieve/rerank, then filter to high-score docs only
    default       → same as original (query_transform → retrieve → rerank)

    Does NOT call _step_critique — caller handles that so chat_history is
    appended exactly once.
    """
    _MIN_GROUNDING_SCORE = 6.0   # minimum rerank score to keep for not_grounded retry

    critique_type = state._critique_type or "incomplete"

    if critique_type == "incomplete":
        print("[RETRY_STRATEGY] type=incomplete → expanding context")
        state = _run_step(state, "query_transform")

        # Retrieve with increased top_k to pull in more candidate docs
        _expanded_top_k = state.num_papers + 2
        queries = state.rewritten_queries if state.rewritten_queries else [state.user_query]
        retriever = RetrieverAgent()
        all_docs = []
        for q in queries:
            all_docs.extend(retriever.retrieve(q, top_k=_expanded_top_k))
        # Dedup (mirrors _step_retrieve logic)
        seen_ids: set = set()
        merged = []
        for doc in all_docs:
            key = doc.get("id") or doc.get("text", "")[:100]
            if key not in seen_ids:
                seen_ids.add(key)
                merged.append(doc)
        state.raw_docs = merged
        print(f"[RETRY] expanded retrieve → {len(state.raw_docs)} docs (top_k={_expanded_top_k})")
        state = _run_step(state, "rerank")

    elif critique_type == "incorrect":
        print("[RETRY_STRATEGY] type=incorrect → tightening relevance")
        state = _run_step(state, "query_transform")
        state = _run_step(state, "retrieve")
        # Rerank with a smaller top_k to keep only the most precise docs
        state.ranked_docs = Reranker().rerank(state.user_query, state.raw_docs, top_k=3)
        print(f"[RETRY] tight rerank → {len(state.ranked_docs)} docs (top_k=3)")

    elif critique_type == "not_grounded":
        print("[RETRY_STRATEGY] type=not_grounded → filtering context")
        state = _run_step(state, "query_transform")
        state = _run_step(state, "retrieve")
        state = _run_step(state, "rerank")
        # Keep only docs that scored above the grounding threshold
        _all_ranked = state.ranked_docs
        state.ranked_docs = [
            d for d in _all_ranked
            if d.get("rerank_score", 0.0) >= _MIN_GROUNDING_SCORE
        ]
        if not state.ranked_docs:
            # Nothing survived the filter — fall back to top-1 to avoid empty context
            state.ranked_docs = _all_ranked[:1]
        print(
            f"[RETRY] grounding filter → {len(state.ranked_docs)} docs "
            f"(threshold={_MIN_GROUNDING_SCORE})"
        )

    else:
        # Default: same behaviour as original (covers "good", unexpected values)
        state = _run_step(state, "query_transform")
        state = _run_step(state, "retrieve")
        state = _run_step(state, "rerank")

    # Force a fresh confidence check so generate_answer doesn't reuse stale score
    state.confidence_cached = False

    state = _run_step(state, "answer")
    state._retried = True
    print(f"[RETRY] done — type={critique_type}  new answer length={len(state.final_answer)}")
    return state


# ---------------------------------------------------------------------------
# Shared retrieval pipeline (Phases 1–2)
# Used by both run_pipeline and run_pipeline_to_context (streaming)
# ---------------------------------------------------------------------------

def run_pipeline_to_context(
    query: str, num_papers: int = 5, chat_history: list = None
) -> State:
    """
    Runs the retrieval pipeline (query_transform → retrieve → [ingest] → rerank).
    Returns State with ranked_docs and sources populated.
    Used by the /stream endpoint before token-by-token answer generation.
    """
    state = State(user_query=query, num_papers=num_papers)
    if chat_history:
        state.chat_history = list(chat_history)

    # Phase 1: probe retrieval with existing index
    state = _run_step(state, "query_transform")
    state = _run_step(state, "retrieve")

    # Phase 2a: multi-signal pre-rerank guard (rerank not yet available)
    _raw_scores = [d.get("score", 0.0) for d in state.raw_docs if d.get("score") is not None]
    _retrieval_norm_2a = sum(_raw_scores) / len(_raw_scores) if _raw_scores else 0.0
    print(f"[INGESTION CHECK] docs found: {len(state.raw_docs)}  retrieval_norm={_retrieval_norm_2a:.3f}")

    _should_ingest, _reason, _strength = should_trigger_ingestion(
        n_docs=len(state.raw_docs),
        retrieval_norm=_retrieval_norm_2a,
        # rerank_scores / rerank_norm / llm_score not yet available — rules that
        # require them are automatically skipped inside should_trigger_ingestion
    )
    state._decision_strength = _strength
    if _should_ingest:
        print(f"[INGESTION TRIGGERED] Phase 2a reason={_reason} strength={_strength} → ingesting")
        for step in INGESTION_STEPS:
            state = _run_step(state, step)
        state = _run_step(state, "retrieve")
        state.ingestion_done = True
        state.decision_trace["action"] = REASON_TO_ACTION[_reason]

    # Phase 2b: multi-signal post-rerank guard (llm_score not yet available)
    state = _run_step(state, "rerank")

    _rerank_scores_2b = [d.get("rerank_score", 0.0) for d in state.ranked_docs if isinstance(d, dict)]
    _rerank_norm_2b   = sum(_rerank_scores_2b) / (10.0 * len(_rerank_scores_2b)) if _rerank_scores_2b else 0.0
    _ret_scores_2b    = [d.get("score", 0.0) for d in state.ranked_docs if d.get("score") is not None]
    _retrieval_norm_2b = sum(_ret_scores_2b) / len(_ret_scores_2b) if _ret_scores_2b else 0.0

    _should_ingest, _reason, _strength = should_trigger_ingestion(
        n_docs=len(state.ranked_docs),
        retrieval_norm=_retrieval_norm_2b,
        rerank_scores=_rerank_scores_2b,
        rerank_norm=_rerank_norm_2b,
        # llm_score not yet available — Rule 4 is automatically skipped
        query=state.rewritten_queries[0] if state.rewritten_queries else state.user_query,
        docs=state.ranked_docs,
    )
    state._decision_strength = _strength
    if _should_ingest:
        print(f"[INGESTION CHECK] Phase 2b reason={_reason} strength={_strength} → triggering ingestion")
        for step in INGESTION_STEPS:
            state = _run_step(state, step)
        state = _run_step(state, "retrieve")
        state = _run_step(state, "rerank")
        state.ingestion_done = True
        state.decision_trace["action"] = REASON_TO_ACTION[_reason]
    else:
        print(f"[INGESTION CHECK] relevance sufficient → skipping ingestion (strength={_strength})")

    # Pre-populate sources from ranked docs (titles + URLs from Pinecone metadata)
    seen: set = set()
    for doc in state.ranked_docs:
        meta = doc.get("metadata", {}) or {}
        title = meta.get("title", "")
        url = meta.get("url", "")
        if title and title not in seen:
            seen.add(title)
            state.sources.append({"title": title, "url": url})

    return state


# ---------------------------------------------------------------------------
# Main entry point (sync, non-streaming)
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str, num_papers: int = 5, chat_history: list = None
) -> tuple[str, float, list, list, dict, str, dict]:
    """
    Returns (answer, confidence, updated_chat_history, sources, latency_ms, status).
    Pass chat_history from the previous call to enable conversational memory.
    """
    # Phases 1–2: retrieval
    state = run_pipeline_to_context(query, num_papers, chat_history)

    # Phase 2c: LLM-based confidence pre-check (only when ingestion hasn't run yet)
    # If confidence is sufficient we cache it so generate_answer doesn't recompute it.
    if not state.ingestion_done:
        context_text = "\n\n".join(
            doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)
        )
        agent = AnswerAgent()
        confidence = agent.get_context_confidence(state.user_query, context_text)
        state.confidence = confidence
        print(f"[ANSWER CHECK] Phase 2c confidence: {confidence:.2f}")

        _rerank_scores_2c = [d.get("rerank_score", 0.0) for d in state.ranked_docs if isinstance(d, dict)]
        _rerank_norm_2c   = sum(_rerank_scores_2c) / (10.0 * len(_rerank_scores_2c)) if _rerank_scores_2c else 0.0
        _ret_scores_2c    = [d.get("score", 0.0) for d in state.ranked_docs if d.get("score") is not None]
        _retrieval_norm_2c = sum(_ret_scores_2c) / len(_ret_scores_2c) if _ret_scores_2c else 0.0

        _should_ingest, _reason, _strength = should_trigger_ingestion(
            n_docs=len(state.ranked_docs),
            retrieval_norm=_retrieval_norm_2c,
            rerank_scores=_rerank_scores_2c,
            rerank_norm=_rerank_norm_2c,
            llm_score=confidence,
            query=state.rewritten_queries[0] if state.rewritten_queries else state.user_query,
            docs=state.ranked_docs,
        )
        state._decision_strength = _strength
        if _should_ingest:
            print(f"[INGESTION CHECK] Phase 2c reason={_reason} → triggering ingestion (retry)")
            for step in INGESTION_STEPS:
                state = _run_step(state, step)
            state = _run_step(state, "retrieve")
            state = _run_step(state, "rerank")
            state.ingestion_done = True
            state.confidence_cached = False
            state.decision_trace["action"] = REASON_TO_ACTION[_reason]
        else:
            # valid for current ranked_docs — reuse in generate_answer (no duplicate call)
            state.confidence_cached = True

    # Phase 3a: generate answer
    state = _run_step(state, "answer")

    # Phase 3b: self-critique score (1 LLM call — small model, max_tokens=80)
    if not state.is_fallback:
        _context_texts = [
            doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)
        ]
        state._critique_score, state._critique_reason, state._critique_type = critique_answer(
            state.user_query, state.final_answer, _context_texts
        )

        # Phase 3c: conditional retry — only on borderline decisions with low scores
        # "weak" decision_strength means signals were ambiguous (borderline case).
        # A single retry is allowed; _retried prevents infinite loops.
        if (
            state._decision_strength == "weak"
            and state._critique_score < 0.6
            and not state._retried
        ):
            _RETRY_DELTA = 0.05   # minimum improvement required to accept retry

            # Snapshot original answer and score before overwriting
            _original_answer = state.final_answer
            _original_score  = state._critique_score
            _original_reason = state._critique_reason
            _original_type   = state._critique_type

            state = _retry_with_expanded_context(state)

            # Re-score after retry so the stored score reflects the final answer
            _context_texts = [
                doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)
            ]
            state._critique_score, state._critique_reason, state._critique_type = critique_answer(
                state.user_query, state.final_answer, _context_texts
            )

            # Accept only if retry is meaningfully better
            if state._critique_score >= _original_score + _RETRY_DELTA:
                print(
                    f"[RETRY_ACCEPTED] improvement {_original_score:.2f} → "
                    f"{state._critique_score:.2f}"
                )
            else:
                print(
                    f"[RETRY_REJECTED] insufficient improvement "
                    f"({_original_score:.2f} → {state._critique_score:.2f}) — reverting"
                )
                state.final_answer     = _original_answer
                state._critique_score  = _original_score
                state._critique_reason = _original_reason
                state._critique_type   = _original_type

    # Phase 3d: refine answer (existing CritiqueAgent) + append to chat_history
    state = _run_step(state, "critique")

    # Composite confidence — replaces the single-signal LLM score with a
    # weighted combination of retrieval similarity, rerank quality, and LLM score.
    # All three signals are already on state; no extra LLM calls.
    compute_composite(state)
    state.status = derive_status(state)

    # Finalise action if not already set to "triggered_ingestion"
    if not state.decision_trace["action"]:
        if state.is_fallback:
            state.decision_trace["action"] = "fallback_no_answer"
        else:
            state.decision_trace["action"] = "used_existing_knowledge"
    print(f"[PIPELINE] status={state.status}  confidence={state.confidence:.3f}")

    return (
        state.final_answer,
        state.confidence,
        state.chat_history,
        state.sources,
        state.latency_ms,
        state.status,
        state.decision_trace,
    )
