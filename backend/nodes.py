import os
import re
import json
from graph_state import GraphState
from agents.tavily_agent import TavilyAgent
from agents.downloader_agent import DownloaderAgent
from agents.embedder_agent import EmbedderAgent
from agents.index_agent import IndexAgent
from agents.text_utils import preprocess_text, chunk_text

tavily = TavilyAgent()
downloader = DownloaderAgent()
embedder = EmbedderAgent()
indexer = IndexAgent()

# 1. SEARCH WEB (respects num_papers)
def node_search_web(state: GraphState) -> GraphState:
    query = state["user_query"]
    num_papers = state.get("num_papers", 5)

    print(f"[SEARCH] Tavily → query='{query}', limit={num_papers}")

    # Get EXACT k results
    papers = tavily.search(query, max_results=num_papers)
    papers = papers[:num_papers]

    state["papers"] = papers
    return state


# 2. OPTIONAL ABSTRACT SUMMARIZATION
def node_summarize_abstracts(state: GraphState) -> GraphState:
    return state


# 3. DOWNLOAD + EXTRACT PDFs
def node_download_and_extract(state: GraphState) -> GraphState:
    papers = state["papers"]
    enriched = []

    for item in papers:
        enriched.append(downloader.download_and_extract(item))

    state["papers"] = enriched
    return state


# 4. TEXT PREPROCESSING
def node_preprocess(state: GraphState) -> GraphState:
    for p in state["papers"]:
        p["clean_text"] = preprocess_text(p.get("full_text", ""))
    return state


# 5. TOKENIZATION (OPTIONAL)
def node_tokenize(state: GraphState) -> GraphState:
    return state


# 6. CHUNK TEXT
def node_chunk(state: GraphState) -> GraphState:
    for p in state["papers"]:
        p["chunks"] = chunk_text(
            p.get("clean_text", ""),
            chunk_size=1000,
            overlap=150
        )
    return state


# 7. EMBED CHUNKS
def node_embed(state: GraphState) -> GraphState:
    for p in state["papers"]:
        p["embeddings"] = embedder.embed_chunks(p["chunks"])
    return state


# 8. INDEX INTO PINECONE
def node_index(state: GraphState) -> GraphState:
    for p in state["papers"]:
        indexer.index_chunks(
            p["title"],
            p["chunks"],
            p["embeddings"]
        )
    return state


# 9. FINAL NODE (MERGES STATE PROPERLY)
def node_finalize(state: GraphState) -> GraphState:
    papers = state.get("papers", [])
    summary = f"Indexed {len(papers)} papers.\n"

    for p in papers:
        summary += (
            f"- {p.get('title', 'Untitled')} → "
            f"{len(p.get('chunks', []))} chunks\n"
        )

    state["final_message"] = summary
    return state
