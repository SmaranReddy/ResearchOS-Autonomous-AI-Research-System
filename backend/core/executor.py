import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
from retrieval.retriever import RetrieverAgent, get_retriever
from retrieval.reranker import Reranker, get_reranker
from agents.answer_agent import AnswerAgent
from agents.critique_agent import CritiqueAgent


# ---------------------------------------------------------------------------
# Latency budget constants
# ---------------------------------------------------------------------------

# Hard cap: if the pipeline exceeds this wall-clock time before Phase 3d,
# skip CritiqueAgent refinement and return the best answer available.
MAX_TOTAL_TIME = 15.0          # seconds

# Skip Phase 3b critique-scoring AND Phase 3d refinement when context
# confidence is already this high — answer quality is sufficient.
# NOTE: this threshold is compared against the RAW LLM confidence score
# (output of get_context_confidence, typically 0.55–0.80 for good context),
# NOT the composite score shown in API responses (which includes rerank signal).
# 0.65 corresponds to composite ~0.80+ in practice.
_EARLY_EXIT_CONF = 0.60

# Rerank median threshold for skipping the Phase 2c LLM confidence call.
# The reranker already scores docs 0–10 for semantic relevance.
# A median >= 5.0 with 3+ docs reliably indicates sufficient context.
_RERANK_CONF_SKIP_THRESHOLD = 5.0

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

# When True: Phase 3b (critique scoring) always runs, enabling the self-critique
# retry loop (Phase 3c) regardless of the early-exit confidence threshold.
# When False: Phase 3b is skipped for high-confidence answers (_EARLY_EXIT_CONF),
# saving ~2 LLM calls per query but disabling retry entirely.
# Trade-off: True = better answer quality on uncertain queries, +2–6 s latency.
#            False = fastest path, retry never fires.
RETRY_ENABLED: bool = True


def _ts() -> str:
    """Current wall-clock timestamp for structured log tags."""
    return time.strftime('%H:%M:%S')


# ---------------------------------------------------------------------------
# Background ingestion — non-blocking
# ---------------------------------------------------------------------------

_ingestion_lock = threading.Lock()
_active_ingestions: set = set()


def _launch_background_ingestion(query: str, num_papers: int) -> None:
    """
    Launch the full ingestion pipeline (search → download → embed → index)
    in a daemon thread so the current request is NOT blocked.
    The next request for the same or related topic will benefit from the
    freshly indexed content.  Deduplicates concurrent runs for the same query.
    """
    with _ingestion_lock:
        if query in _active_ingestions:
            print(f"[BG_INGESTION] Already running for: '{query[:60]}'")
            return
        _active_ingestions.add(query)

    def _run() -> None:
        try:
            print(f"[BG_INGESTION] Started: '{query[:60]}'")
            bg_state = State(user_query=query, num_papers=num_papers)
            for step in INGESTION_STEPS:
                bg_state = STEP_REGISTRY[step](bg_state)
            print(f"[BG_INGESTION] Complete: '{query[:60]}'")
        except Exception as exc:
            print(f"[BG_INGESTION] Error: {exc}")
        finally:
            with _ingestion_lock:
                _active_ingestions.discard(query)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


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
        full_text = p.get("full_text", "")
        # Fall back to Tavily abstract/snippet when PDF download failed
        if not full_text.strip():
            fallback = p.get("abstract") or p.get("summary") or p.get("snippet") or ""
            if fallback.strip():
                print(f"  [preprocess] PDF empty — using abstract for '{p.get('title', 'unknown')}'")
                full_text = fallback
        print(f"  [preprocess] processing '{p.get('title', 'unknown')}'")
        p["clean_text"] = preprocessor.preprocess(full_text)
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
    # rewritten_queries[0] is always the standalone resolved query
    state.resolved_query = state.rewritten_queries[0] if state.rewritten_queries else state.user_query
    return state


_RETRIEVAL_TOP_K = 5          # max docs per query
_RETRIEVAL_TIMEOUT_S = 5.0    # per-Pinecone-call timeout (generous for cold start)
_MAX_QUERIES = 3              # hard cap on fan-out


def _step_retrieve(state: State) -> State:
    queries = state.rewritten_queries if state.rewritten_queries else [state.user_query]
    # Hard cap — defensive guard regardless of how many queries were generated
    queries = queries[:_MAX_QUERIES]

    # Confirmation log — if this does NOT appear in server output, the old
    # cached process is still running.  Restart the server to pick up new code.
    print(f"[PARALLEL_RETRIEVE_ACTIVE]")
    print(f"[NUM_QUERIES] {len(queries)}")
    print(f"[RETRIEVE_START] querying Pinecone with {len(queries)} queries")

    retriever = get_retriever()  # reuse singleton — avoids recreating Pinecone client
    # retrieve_many: embeds queries sequentially (one GIL-safe pass on the
    # shared SentenceTransformer model), then fires all Pinecone calls in
    # parallel (pure network I/O — GIL released, truly concurrent).
    all_docs = retriever.retrieve_many(
        queries, top_k=_RETRIEVAL_TOP_K, timeout_s=_RETRIEVAL_TIMEOUT_S
    )

    print(f"[DOCS_FETCHED] {len(all_docs)} before dedup")

    # Deduplicate by chunk id (or first 100 chars of text as fallback)
    seen_ids: set = set()
    merged: list = []
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
    # Use the resolved (standalone) query so reranking is context-aware,
    # e.g. "compare it to pure transformer" → "Compare BERT and GPT to pure Transformer"
    rerank_query = state.resolved_query or state.user_query
    # get_reranker() returns the module-level singleton — reuses the Groq
    # httpx connection pool instead of discarding it on every call.
    state.ranked_docs = get_reranker().rerank(rerank_query, state.raw_docs)
    return state


def _step_answer(state: State) -> State:
    print("[ANSWER_START] generating final answer")
    # Use the resolved standalone query for confidence scoring and prompt construction.
    # This ensures "compare it to pure transformer" → "Compare BERT and GPT to pure Transformer"
    # so the LLM and confidence gate operate on an unambiguous question.
    effective_query = state.resolved_query or state.user_query
    state.final_answer = AnswerAgent().generate_answer(
        effective_query, state.ranked_docs, chat_history=state.chat_history, state=state
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
    _tag = step.upper()
    print(f"\n[{_tag}_START] {_ts()}")
    t0 = time.monotonic()
    state = STEP_REGISTRY[step](state)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    print(f"[{_tag}_TIME] {elapsed_ms}ms")

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
        print(f"[STEP] retrieve → {len(state.raw_docs)} docs retrieved")
    elif step == "rerank":
        print(f"[STEP] rerank → top {len(state.ranked_docs)} docs")
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
        queries = (state.rewritten_queries if state.rewritten_queries else [state.user_query])[:_MAX_QUERIES]
        retriever = get_retriever()  # reuse singleton
        all_docs = retriever.retrieve_many(queries, top_k=_expanded_top_k, timeout_s=_RETRIEVAL_TIMEOUT_S)
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
        rerank_query = state.resolved_query or state.user_query
        state.ranked_docs = Reranker().rerank(rerank_query, state.raw_docs, top_k=3)
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
    query: str, num_papers: int = 3, chat_history: list = None
) -> State:
    """
    Runs the retrieval pipeline (query_transform → retrieve → [ingest] → rerank).
    Returns State with ranked_docs and sources populated.
    Used by the /stream endpoint before token-by-token answer generation.
    """
    _t_ctx_start = time.monotonic()
    state = State(user_query=query, num_papers=num_papers)
    if chat_history:
        state.chat_history = list(chat_history)

    # Phase 1: query_transform + initial Pinecone probe run in parallel.
    # QueryTransformer calls the Groq API (pure I/O — GIL released), so the
    # probe's embed+Pinecone runs concurrently, hiding transform latency behind
    # retrieve latency.  Wall time ≈ max(transform, probe) instead of sum.
    print(f"\n[QUERY_TRANSFORM_START] {_ts()}")
    print(f"\n[RETRIEVE_START] {_ts()}")
    _t_retrieve = time.monotonic()

    retriever = get_retriever()
    with ThreadPoolExecutor(max_workers=2) as _pool:
        _transform_fut = _pool.submit(
            QueryTransformer().transform,
            state.user_query,
            state.chat_history if state.chat_history else None,
        )
        # Probe with original query while Groq generates variations
        _probe_fut = _pool.submit(retriever.retrieve, state.user_query, _RETRIEVAL_TOP_K)

        try:
            _rewritten = _transform_fut.result(timeout=6.0)
        except Exception as _e:
            print(f"[WARN] query_transform failed ({_e}) — using original query")
            _rewritten = [state.user_query]

        try:
            _probe_docs = _probe_fut.result(timeout=_RETRIEVAL_TIMEOUT_S)
        except Exception:
            _probe_docs = []

    state.rewritten_queries = (_rewritten or [state.user_query])[:_MAX_QUERIES]
    state.resolved_query = state.rewritten_queries[0]
    print(f"[PARALLEL_RETRIEVE_ACTIVE]")
    print(f"[MULTI-QUERY] {len(state.rewritten_queries)} queries (parallel transform+probe)")

    # Fire variation queries (original already probed above)
    _variations = state.rewritten_queries[1:]
    _variation_docs = (
        retriever.retrieve_many(_variations, top_k=_RETRIEVAL_TOP_K, timeout_s=_RETRIEVAL_TIMEOUT_S)
        if _variations else []
    )

    # Merge + dedup
    _all_raw = (_probe_docs or []) + _variation_docs
    _seen: set = set()
    _merged: list = []
    for _doc in _all_raw:
        _key = _doc.get("id") or _doc.get("text", "")[:100]
        if _key not in _seen:
            _seen.add(_key)
            _merged.append(_doc)
    state.raw_docs = _merged

    _retrieve_ms = int((time.monotonic() - _t_retrieve) * 1000)
    state.latency_ms["retrieve_ms"] = state.latency_ms.get("retrieve_ms", 0) + _retrieve_ms
    print(f"[NUM_QUERIES] {len(state.rewritten_queries)}")
    print(f"[DOCS_FETCHED] {len(_all_raw)} before dedup")
    print(f"[RETRIEVE] after dedup: {len(_merged)}")
    print(f"[RETRIEVE_TIME] {_retrieve_ms}ms")
    print(f"[STEP] retrieve → {len(state.raw_docs)} docs retrieved")

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
        print(f"[INGESTION TRIGGERED] Phase 2a reason={_reason} strength={_strength} → background")
        # Non-blocking: ingest in background, answer with existing docs immediately.
        # The next request benefits from freshly indexed content.
        _launch_background_ingestion(state.user_query, state.num_papers)
        state.ingestion_done = True  # prevents Phase 2b/2c from double-triggering
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
    if _should_ingest and not state.ingestion_done:
        print(f"[INGESTION CHECK] Phase 2b reason={_reason} strength={_strength} → background")
        # Non-blocking: ingest in background, answer with current ranked docs.
        _launch_background_ingestion(state.user_query, state.num_papers)
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
    query: str, num_papers: int = 3, chat_history: list = None, disable_retry: bool = False
) -> tuple[str, float, list, list, dict, str, dict]:
    """
    Returns (answer, confidence, updated_chat_history, sources, latency_ms, status).
    Pass chat_history from the previous call to enable conversational memory.
    """
    _t_pipeline_start = time.monotonic()
    print(f"\n[TOTAL_START] {_ts()}")

    # Phases 1–2: retrieval
    state = run_pipeline_to_context(query, num_papers, chat_history)

    # Phase 2c: confidence pre-check — decide whether context needs fresh ingestion.
    # Fast path: derive confidence from rerank scores without an extra LLM call.
    # The reranker already scored every doc 0–10 for semantic relevance; a median
    # >= _RERANK_CONF_SKIP_THRESHOLD with 3+ docs is a reliable proxy for sufficient
    # context and eliminates one round-trip to the Groq API (~1–3 s saved per query).
    if not state.ingestion_done:
        print(f"\n[CONF_CHECK_START] {_ts()}")
        _t2c = time.monotonic()

        _rerank_scores_2c = [d.get("rerank_score", 0.0) for d in state.ranked_docs if isinstance(d, dict)]
        _rerank_norm_2c   = sum(_rerank_scores_2c) / (10.0 * len(_rerank_scores_2c)) if _rerank_scores_2c else 0.0
        _ret_scores_2c    = [d.get("score", 0.0) for d in state.ranked_docs if d.get("score") is not None]
        _retrieval_norm_2c = sum(_ret_scores_2c) / len(_ret_scores_2c) if _ret_scores_2c else 0.0

        # Median rerank score — more robust than mean against outliers
        _rs_sorted  = sorted(_rerank_scores_2c, reverse=True)
        _rerank_med = _rs_sorted[len(_rs_sorted) // 2] if _rs_sorted else 0.0
        _good_context = (len(state.ranked_docs) >= 3 and _rerank_med >= _RERANK_CONF_SKIP_THRESHOLD)

        if _good_context:
            # Estimate confidence from rerank median (no LLM call needed).
            # Formula maps median score [5, 10] → confidence [0.68, 0.90].
            confidence = min(0.63 + _rerank_med * 0.027, 0.90)
            state.confidence = confidence
            state.confidence_cached = True
            print(
                f"[CONF_CHECK_TIME] 0ms  confidence={confidence:.2f}  "
                f"(rerank-based, skipped LLM call — median={_rerank_med:.1f} docs={len(state.ranked_docs)})"
            )
        else:
            # Context looks weak — use LLM to get an accurate confidence score
            context_text = "\n\n".join(
                doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)
            )
            agent = AnswerAgent()
            _effective_query = state.resolved_query or state.user_query
            confidence = agent.get_context_confidence(_effective_query, context_text)
            state.confidence = confidence
            print(
                f"[CONF_CHECK_TIME] {int((time.monotonic()-_t2c)*1000)}ms  "
                f"confidence={confidence:.2f}  (LLM call — weak rerank median={_rerank_med:.1f})"
            )

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
            # confidence_cached already set in fast path; set here for the slow path too
            state.confidence_cached = True

    # Phase 3a: generate answer
    state = _run_step(state, "answer")

    # Phase 3b: self-critique score (1 LLM call — small model, max_tokens=100)
    # Early-exit: skip Phase 3b (and 3d) when confidence is already high.
    # If confidence >= _EARLY_EXIT_CONF the context is already sufficient;
    # the critique score would not change the retry decision (retry requires
    # score < 0.7, but high-confidence answers almost never get scored that low).
    # Saving this call reduces from 4→3 Groq calls for the common case, which
    # keeps bulk evaluation within the 30 RPM free-tier limit.
    # Retry still fires for low-confidence queries (confidence < _EARLY_EXIT_CONF).
    _skip_3b = (
        not state.is_fallback
        and state.confidence >= _EARLY_EXIT_CONF
    )
    if _skip_3b:
        print(
            f"[CRITIQUE_SCORE] Phase 3b skipped — high confidence "
            f"({state.confidence:.2f} >= {_EARLY_EXIT_CONF})"
        )
        state._critique_score  = 1.0
        state._critique_type   = "good"
        state._critique_reason = "early-exit: high confidence"
    elif not state.is_fallback:
        _context_texts = [
            doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)
        ]
        print(f"\n[CRITIQUE_SCORE_START] {_ts()}")
        _t3b = time.monotonic()
        state._critique_score, state._critique_reason, state._critique_type = critique_answer(
            state.user_query, state.final_answer, _context_texts
        )
        print(f"[CRITIQUE_SCORE_TIME] {int((time.monotonic()-_t3b)*1000)}ms  score={state._critique_score:.2f}")

        # [RETRY_DEBUG] — log all gate variables before the retry condition
        # NOTE: _decision_strength is intentionally NOT part of the gate.
        # It reflects retrieval uncertainty, not answer quality. For well-indexed
        # queries strength is always "strong", which would permanently block retry.
        # The critique score is the correct signal for whether the answer needs improvement.
        # Time budget gate: skip retry if the pipeline is already near the ceiling.
        # A retry costs another query_transform + retrieve + rerank + answer (~5-10s).
        # If we're already past (MAX_TOTAL_TIME - 5s), there's no budget for that.
        _elapsed_pre_retry = time.monotonic() - _t_pipeline_start
        _skip_3c_budget = _elapsed_pre_retry >= MAX_TOTAL_TIME - 5.0

        _gate_open = (
            RETRY_ENABLED
            and not disable_retry
            and state._critique_score < 0.7
            and not state._retried
            and not _skip_3c_budget
        )
        print(
            f"[RETRY_DEBUG] RETRY_ENABLED={RETRY_ENABLED}  disable_retry={disable_retry}  "
            f"elapsed={_elapsed_pre_retry:.1f}s  budget_ok={not _skip_3c_budget}  "
            f"decision_strength={state._decision_strength!r}  "
            f"critique_score={state._critique_score:.2f}  "
            f"critique_type={state._critique_type!r}  "
            f"retried={state._retried}  gate_open={_gate_open}"
        )
        if _skip_3c_budget:
            print(f"[RETRY_SKIP] time budget ({_elapsed_pre_retry:.1f}s / {MAX_TOTAL_TIME}s max)")

        # Phase 3c: conditional retry — fires when answer quality is poor (score < 0.7).
        # Gated by RETRY_ENABLED (module flag), disable_retry (per-request override),
        # and the time budget check above.  A single retry is allowed; _retried prevents loops.
        if (
            RETRY_ENABLED
            and not disable_retry
            and state._critique_score < 0.7
            and not state._retried
            and not _skip_3c_budget
        ):
            _RETRY_DELTA = 0.05   # minimum improvement required to accept retry

            print(
                f"\n[RETRY_START] {_ts()}  reason={state._critique_type!r}  "
                f"strength={state._decision_strength!r}  score={state._critique_score:.2f}"
            )
            _t_retry = time.monotonic()

            # Snapshot original answer and score before overwriting
            _original_answer = state.final_answer
            _original_score  = state._critique_score
            _original_reason = state._critique_reason
            _original_type   = state._critique_type

            state = _retry_with_expanded_context(state)

            print(f"[RETRY_TIME] {int((time.monotonic()-_t_retry)*1000)}ms")

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

    # Phase 3d: CritiqueAgent refinement — skip if time budget exhausted or high confidence
    _elapsed_before_3d = time.monotonic() - _t_pipeline_start
    _skip_3d_budget     = _elapsed_before_3d >= MAX_TOTAL_TIME - 2.0
    _skip_3d_confidence = state.confidence >= _EARLY_EXIT_CONF

    if _skip_3d_budget:
        print(
            f"[CRITIQUE] Phase 3d skipped — time budget "
            f"({_elapsed_before_3d:.1f}s / {MAX_TOTAL_TIME}s max)"
        )
        # Still append to history so conversational memory works
        state.chat_history.append({"query": state.user_query, "answer": state.final_answer})
        state.chat_history = state.chat_history[-3:]
    elif _skip_3d_confidence:
        print(
            f"[CRITIQUE] Phase 3d skipped — high confidence "
            f"({state.confidence:.2f} >= {_EARLY_EXIT_CONF})"
        )
        state.chat_history.append({"query": state.user_query, "answer": state.final_answer})
        state.chat_history = state.chat_history[-3:]
    else:
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
    state.latency_ms["total_ms"] = int((time.monotonic() - _t_pipeline_start) * 1000)
    print(f"[RETRIEVE_TIME] {state.latency_ms['retrieve_ms']}ms")
    print(f"[RERANK_TIME]   {state.latency_ms['rerank_ms']}ms")
    print(f"[LLM_TIME]      {state.latency_ms['llm_ms']}ms")
    print(f"[TOTAL_TIME]    {state.latency_ms['total_ms']}ms")
    print(f"[PIPELINE] status={state.status}  confidence={state.confidence:.3f}")

    return (
        state.final_answer,
        state.confidence,
        state.chat_history,
        state.sources,
        state.latency_ms,
        state.status,
        state.decision_trace,
        state._retried,
    )
