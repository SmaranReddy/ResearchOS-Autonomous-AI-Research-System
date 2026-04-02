from core.state import State
from core.planner import get_plan

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
        p["clean_text"] = preprocessor.preprocess(p.get("full_text", ""))
    return state


def _step_chunk(state: State) -> State:
    print("[chunk] splitting text into chunks")
    chunker = Chunker(chunk_size=1000, overlap=150)
    for p in state.papers:
        p["chunks"] = chunker.chunk_text(p.get("clean_text", ""))
    return state


def _step_embed(state: State) -> State:
    print("[embed] embedding chunks")
    embedder = Embedder()
    for p in state.papers:
        p["embeddings"] = embedder.embed_chunks(p.get("chunks", []))
    return state


def _step_index(state: State) -> State:
    print("[index] uploading to Pinecone")
    indexer = Indexer()
    for p in state.papers:
        indexer.index_chunks(p["title"], p.get("chunks", []), p.get("embeddings", []))
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
        state.user_query, state.ranked_docs, chat_history=state.chat_history
    )
    return state


def _step_critique(state: State) -> State:
    print("[CRITIQUE] refining answer")
    context = [doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)]
    state.final_answer = CritiqueAgent().critique(state.final_answer, context, query=state.user_query)
    # append refined answer to history and keep only last 3
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
# Retrieval quality check
# ---------------------------------------------------------------------------

MIN_DOCS = 3
MIN_AVG_SCORE = 0.3

INGESTION_STEPS = ["search_web", "download", "preprocess", "chunk", "embed", "index"]


def _is_retrieval_weak(docs: list) -> bool:
    if len(docs) < MIN_DOCS:
        return True
    scores = [d.get("score") for d in docs if d.get("score") is not None]
    if scores and (sum(scores) / len(scores)) < MIN_AVG_SCORE:
        return True
    return False


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

    # --- Phase 2: evaluate retrieval quality ---
    print(f"[INGESTION CHECK] docs found: {len(state.raw_docs)}")
    if _is_retrieval_weak(state.raw_docs):
        print(f"[INGESTION TRIGGERED] due to low retrieval ({len(state.raw_docs)} docs)")
        for step in INGESTION_STEPS:
            state = _run_step(state, step)
        # re-retrieve with freshly indexed content
        state = _run_step(state, "retrieve")
    else:
        print(f"[INGESTION SKIPPED] sufficient docs found ({len(state.raw_docs)})")

    # --- Phase 3: rerank + answer + critique ---
    state = _run_step(state, "rerank")
    state = _run_step(state, "answer")
    state = _run_step(state, "critique")

    return state.final_answer, state.chat_history
