# ==========================================
# backend/pipeline.py — Final Interactive Research Assistant
# ==========================================

import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph
import nodes
from agents.retriever_agent import RetrieverAgent
from agents.index_agent import IndexAgent
from agents.critique_agent import CritiqueAgent

# ------------------------------------------------
# STATE SCHEMA (Required by latest LangGraph)
# ------------------------------------------------
class GraphState(TypedDict):
    user_query: str
    num_papers: int
    papers: List[Dict[str, Any]]

# ------------------------------------------------
# DYNAMIC NODE LOADER (auto-detects names)
# ------------------------------------------------
def get_node(name_variants):
    for n in name_variants:
        if hasattr(nodes, n):
            return getattr(nodes, n)
    raise ImportError(f"No matching node found for {name_variants}")

# Load node functions dynamically
node_search_web = get_node(["node_search_web", "node_search"])
node_download_and_extract = get_node(["node_download_and_extract", "node_download"])
node_clean = get_node(["node_clean", "node_preprocess", "node_clean_text"])
node_chunk = get_node(["node_chunk"])
node_embed = get_node(["node_embed"])
node_index = get_node(["node_index"])

# ------------------------------------------------
# SETUP GRAPH PIPELINE
# ------------------------------------------------
_graph = StateGraph(GraphState)
_graph.add_node("search_web", node_search_web)
_graph.add_node("download_and_extract", node_download_and_extract)
_graph.add_node("clean", node_clean)
_graph.add_node("chunk", node_chunk)
_graph.add_node("embed", node_embed)
_graph.add_node("index", node_index)

_graph.set_entry_point("search_web")
_graph.add_edge("search_web", "download_and_extract")
_graph.add_edge("download_and_extract", "clean")
_graph.add_edge("clean", "chunk")
_graph.add_edge("chunk", "embed")
_graph.add_edge("embed", "index")
_graph.set_finish_point("index")

_graph = _graph.compile()

# ------------------------------------------------
# MAIN PIPELINE FUNCTION
# ------------------------------------------------
def run_pipeline(query: str, num_papers: int = 5):
    print(f"\n🚀 Starting indexing phase for query: '{query}'")
    state = _graph.invoke({"user_query": query, "num_papers": num_papers})

    papers = state.get("papers", [])
    print(f"\n✅ Indexed {len(papers)} papers.")
    for p in papers:
        print(f"- {p['title']} → {len(p.get('chunks', []))} chunks")

    print("\n🔍 Moving to retrieval and summarization phase...\n")

    retriever = RetrieverAgent()
    answerer = AnswerAgent()
    critiquer = CritiqueAgent()

    # Step 1: Retrieve relevant chunks
    retrieved = retriever.retrieve(query)
    print(f"✅ Retrieved {len(retrieved)} chunks for answer synthesis.\n")

    # Step 2: Generate raw answer
    raw_answer = answerer.generate_answer(query, retrieved)
    print("🧠 Raw Answer Generated.\n")

    # Step 3: Critique/refine the answer
    refined = critiquer.critique(raw_answer)
    print("\n✅ Refined Final Research Summary:\n")
    print(refined[:10000])  # Show up to 10k chars for long summaries

    # Step 4: Enter interactive Q&A
    print("\n🗣️ Now you can ask any questions related to this topic.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_q = input("\n💬 Ask your question: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("\n👋 Exiting interactive research session.")
            break

        retrieved_q = retriever.retrieve(user_q)
        print(f"✅ Retrieved {len(retrieved_q)} context chunks.\n")

        raw_q_answer = answerer.generate_answer(user_q, retrieved_q)
        refined_q_answer = critiquer.critique(raw_q_answer)

        print("\n🤖 Answer:\n", refined_q_answer[:5000])

# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":
    q = input("Enter your research query: ").strip()
    run_pipeline(q, num_papers=5)
