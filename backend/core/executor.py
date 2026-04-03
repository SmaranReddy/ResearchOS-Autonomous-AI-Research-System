from core.state import State

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
# Each function receives State, calls one service, updates state, returns State.
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
    print("[index] uploading to Pinecone")
    indexer = Indexer()
    for p in state.papers:
        if p.get("indexed"):
            print(f"  [index] skip '{p.get('title', 'unknown')}' — already indexed")
            continue
        print(f"  [index] indexing '{p.get('title', 'unknown')}'")
        indexer.index_chunks(p["title"], p.get("chunks", []), p.get("embeddings", []))
        p["indexed"] = True
    return state


def _step_query_transform(state: State) -> State:
    print("[query_transform] generating multi-query variations")
    state.rewritten_queries = QueryTransformer().transform(state.user_query)
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
# Step registry — maps plan step name → implementation
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


# ---------------------------------------------------------------------------
# Retrieval quality checks
# ---------------------------------------------------------------------------

MIN_DOCS = 3
MIN_AVG_SCORE = 0.3
MIN_RERANK_SCORE = 4.0   # rerank scores are 0–10 (LLM-rated)

INGESTION_STEPS = ["search_web", "download", "preprocess", "chunk", "embed", "index"]


def _is_retrieval_weak(docs: list) -> bool:
    """Fast pre-rerank guard: triggers on very low doc count or poor raw scores."""
    if len(docs) < MIN_DOCS:
        return True
    scores = [d.get("score") for d in docs if d.get("score") is not None]
    if scores and (sum(scores) / len(scores)) < MIN_AVG_SCORE:
        return True
    return False


def _is_relevance_low(ranked_docs: list) -> bool:
    """
    Post-rerank relevance guard.
    Triggers ingestion when the median score is low OR the majority of
    docs are weak — robust to overall weak retrieval without depending
    on single outliers.
    """
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
    state = STEP_REGISTRY[step](state)
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
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(query: str, num_papers: int = 5, chat_history: list = None) -> tuple[str, list]:
    """
    Returns (answer, updated_chat_history).
    Pass chat_history from the previous call to enable conversational memory.
    """
    state = State(user_query=query, num_papers=num_papers)
    if chat_history:
        state.chat_history = list(chat_history)

    # --- Phase 1: probe retrieval with existing index ---
    state = _run_step(state, "query_transform")
    state = _run_step(state, "retrieve")

    # --- Phase 2a: fast count guard (before rerank) ---
    print(f"[INGESTION CHECK] docs found: {len(state.raw_docs)}")
    if _is_retrieval_weak(state.raw_docs):
        print(f"[INGESTION TRIGGERED] too few / low-scoring raw docs ({len(state.raw_docs)}) → ingesting")
        for step in INGESTION_STEPS:
            state = _run_step(state, step)
        state = _run_step(state, "retrieve")
        state.ingestion_done = True
        print(f"[INGESTION CHECK] re-retrieved {len(state.raw_docs)} docs after ingestion")

    # --- Phase 2b: rerank, then check relevance quality ---
    state = _run_step(state, "rerank")

    if _is_relevance_low(state.ranked_docs):
        print("[INGESTION CHECK] relevance low → triggering ingestion")
        for step in INGESTION_STEPS:
            state = _run_step(state, step)
        state = _run_step(state, "retrieve")
        state = _run_step(state, "rerank")
        state.ingestion_done = True
        print("[INGESTION CHECK] relevance sufficient → continuing after re-ingestion")
    else:
        print("[INGESTION CHECK] relevance sufficient → skipping ingestion")

    # --- Phase 2c: confidence pre-check (max 1 retry) ---
    if not state.ingestion_done:
        context_text = "\n\n".join(
            doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)
        )
        agent = AnswerAgent()
        confidence = agent.get_context_confidence(state.user_query, context_text)
        state.confidence = confidence
        print(f"[ANSWER CHECK] confidence: {confidence:.2f}")
        if confidence < agent.CONFIDENCE_THRESHOLD:
            print("[INGESTION CHECK] context insufficient → triggering ingestion (retry)")
            for step in INGESTION_STEPS:
                state = _run_step(state, step)
            state = _run_step(state, "retrieve")
            state = _run_step(state, "rerank")
            state.ingestion_done = True

    # --- Phase 3: answer + critique ---
    state = _run_step(state, "answer")
    state = _run_step(state, "critique")

    return state.final_answer, state.confidence, state.chat_history
