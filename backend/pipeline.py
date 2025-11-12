# backend/pipeline.py
from langgraph.graph import StateGraph, END
from graph_state import GraphState

from nodes import (
    node_search_web,
    node_summarize_abstracts,
    node_download_and_extract,
    node_preprocess,
    node_tokenize,
    node_chunk,
    node_embed,
    node_index,
    node_finalize,
)

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("search_web", node_search_web)
    g.add_node("summarize", node_summarize_abstracts)
    g.add_node("download", node_download_and_extract)
    g.add_node("preprocess", node_preprocess)
    g.add_node("tokenize", node_tokenize)
    g.add_node("chunk", node_chunk)
    g.add_node("embed", node_embed)
    g.add_node("index", node_index)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("search_web")
    g.add_edge("search_web", "summarize")
    g.add_edge("summarize", "download")
    g.add_edge("download", "preprocess")
    g.add_edge("preprocess", "tokenize")
    g.add_edge("tokenize", "chunk")
    g.add_edge("chunk", "embed")
    g.add_edge("embed", "index")
    g.add_edge("index", "finalize")
    g.add_edge("finalize", END)

    return g.compile()

_graph = None

def run_pipeline(query: str, num_papers: int = 3):
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph.invoke({"user_query": query, "num_papers": num_papers})

if __name__ == "__main__":
    q = input("Enter query: ")
    n = input("Number of papers to retrieve: ")
    out = run_pipeline(q, int(n))
    print(out["final_message"])
