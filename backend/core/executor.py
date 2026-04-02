from core.state import State
from core.planner import get_plan

from pinecone import Pinecone
import os
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
    print("[query_transform] rewriting query")
    state.rewritten_query = QueryTransformer().transform(state.user_query)
    return state


def _step_retrieve(state: State) -> State:
    print("[retrieve] querying Pinecone with rewritten query")
    state.raw_docs = RetrieverAgent().retrieve(state.rewritten_query)
    return state


def _step_rerank(state: State) -> State:
    print(f"[rerank] scoring {len(state.raw_docs)} docs")
    state.ranked_docs = Reranker().rerank(state.user_query, state.raw_docs)
    return state


def _step_answer(state: State) -> State:
    print("[answer] generating final answer")
    state.final_answer = AnswerAgent().generate_answer(state.user_query, state.ranked_docs)
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
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

INGESTION_STEPS = {"search_web", "download", "preprocess", "chunk", "embed", "index"}


def _index_has_vectors() -> bool:
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        stats = pc.Index("re-search").describe_index_stats()
        return (stats.get("total_vector_count") or 0) > 0
    except Exception:
        return False


def run_pipeline(query: str, num_papers: int = 5) -> str:
    state = State(user_query=query, num_papers=num_papers)
    plan = get_plan()

    skip_ingestion = _index_has_vectors()
    if skip_ingestion:
        print("[INFO] Pinecone index has vectors — skipping ingestion steps.")

    for step in plan:
        if skip_ingestion and step in INGESTION_STEPS:
            print(f"[SKIP] {step}")
            continue
        print(f"\n[STEP] {step}")
        state = STEP_REGISTRY[step](state)

        # post-step summary logs
        if step == "search_web":
            print(f"[STEP] search_web → {len(state.papers)} papers found")
        elif step == "chunk":
            total = sum(len(p.get("chunks", [])) for p in state.papers)
            print(f"[STEP] chunk → {total} total chunks")
        elif step == "retrieve":
            print(f"[STEP] retrieve → {len(state.raw_docs)} docs retrieved")
        elif step == "rerank":
            print(f"[STEP] rerank → top {len(state.ranked_docs)} docs")

    return state.final_answer
