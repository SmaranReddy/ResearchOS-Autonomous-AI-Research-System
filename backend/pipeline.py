# ==========================================
# backend/pipeline.py
# Full Research Assistant Pipeline
# ==========================================

import os
import sys

# ✅ Ensure backend package imports work
sys.path.append(os.path.dirname(__file__))

from agents.answer_agent import AnswerAgent
from agents.retriever_agent import RetrieverAgent
from agents.index_agent import IndexAgent
from agents.critique_agent import CritiqueAgent
from utils.tavily_utils import TavilyClient


# -------------------------------
# Environment Setup
# -------------------------------
os.environ["TAVILY_API_KEY"] = "<YOUR_TAVILY_API_KEY>"
os.environ["GROQ_API_KEY"] = "<YOUR_GROQ_API_KEY>"
os.environ["PINECONE_API_KEY"] = "<YOUR_PINECONE_API_KEY>"

# -------------------------------
# Main Pipeline
# -------------------------------
def run_pipeline(query: str, num_papers: int = 5):
    print(f"\nEnter your research query: {query}\n")

    # --- Initialize Agents ---
    tavily = TavilyClient()
    indexer = IndexAgent()
    retriever = RetrieverAgent()
    answerer = AnswerAgent()
    critic = CritiqueAgent()

    # --- Step 1: Tavily Search ---
    print(f"🚀 Starting indexing phase for query: '{query}'")
    papers = tavily.search_and_download(query, num_papers=num_papers)
    if not papers:
        print("⚠️ No papers found. Proceeding with overview generation only.\n")

    # --- Step 2: Index into Pinecone ---
    indexer.index_papers(papers)
    print("✅ Indexing completed.\n")

    # --- Step 3: Retrieve Similar Context ---
    print("🔍 Moving to retrieval and summarization phase...\n")
    retrieved_chunks = retriever.retrieve(query)

    # --- Step 4: Generate Academic Overview ---
    overview = answerer.generate_overview(query, retrieved_chunks)

    # --- Step 5: Critique and Refine ---
    reviewed_summary = critic.critique(overview)
    print("\n✅ Refined Final Research Summary:\n")
    print(reviewed_summary)

    # --- Step 6: Interactive Q&A ---
    print("\n🗣️ Now you can ask any questions related to this topic.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("💬 Ask your question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting interactive mode.")
            break
        answer = answerer.generate_precise_answer(question, context_summary=overview)
        print(f"\n🤖 Answer:\n{answer}\n")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    q = input("Enter your research topic: ")
    run_pipeline(q, num_papers=5)
