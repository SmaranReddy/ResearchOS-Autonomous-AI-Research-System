# ==========================================
# frontend/app.py — Professional Streamlit UI
# ==========================================

import os
import sys

# ------------------------------------------
# Ensure project root is importable
# ------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import streamlit as st

from core.executor import run_pipeline
from retrieval.retriever import RetrieverAgent
from agents.answer_agent import AnswerAgent
from agents.critique_agent import CritiqueAgent


# ==========================================
# Streamlit Page Configuration
# ==========================================
st.set_page_config(
    page_title="Research Assistant",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .title {
        font-size: 34px;
        font-weight: 700;
        padding-bottom: 8px;
        margin-bottom: 25px;
        border-bottom: 1px solid #CCC;
    }
    .section-header {
        font-size: 22px;
        font-weight: 600;
        margin-top: 35px;
        margin-bottom: 8px;
    }
    .sub-header {
        font-size: 18px;
        font-weight: 500;
        margin-top: 25px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Header
# ==========================================
st.markdown('<div class="title">Research Assistant</div>', unsafe_allow_html=True)


# ==========================================
# Sidebar Controls
# ==========================================
with st.sidebar:
    st.header("Configuration")

    query = st.text_area("Research Query", height=120)

    num_papers = st.number_input(
        "Number of papers to index",
        min_value=1,
        max_value=50,
        value=5
    )

    run_button = st.button("Run Pipeline")

    st.markdown("---")
    st.header("Follow-up Questions")
    followup_query = st.text_area("Ask a follow-up question")
    followup_button = st.button("Submit Question")


# ==========================================
# Main Execution
# ==========================================
if run_button:

    if not query.strip():
        st.error("Please provide a valid research query.")
        st.stop()

    st.markdown('<div class="section-header">Indexing Papers</div>', unsafe_allow_html=True)

    progress = st.progress(0)
    time.sleep(0.1)

    try:
        # Run indexing (search → download → preprocess → chunk → embed → index)
        run_pipeline(query, num_papers=num_papers)
        progress.progress(100)
        st.success("Indexing phase completed.")
    except Exception as e:
        st.error(f"Pipeline execution failed: {e}")
        st.stop()

    # -----------------------------------------------------
    # Retrieval + Answer Synthesis Phase
    # -----------------------------------------------------
    st.markdown('<div class="section-header">Retrieval and Answer Generation</div>', unsafe_allow_html=True)

    retriever = RetrieverAgent()
    answerer = AnswerAgent()
    critiquer = CritiqueAgent()

    st.write("Retrieving relevant chunks...")
    retrieved = retriever.retrieve(query)
    st.write(f"{len(retrieved)} chunks retrieved.")

    # Show retrieved chunks
    with st.expander("View Retrieved Chunks"):
        for idx, m in enumerate(retrieved, start=1):
            st.markdown(f"**Chunk {idx} — Score {m['score']:.4f}**")
            st.write(f"Title: {m['metadata'].get('title', '')}")
            st.write(m["text"])
            st.markdown("---")

    # Raw answer
    st.markdown('<div class="sub-header">Initial Model Answer</div>', unsafe_allow_html=True)
    raw_answer = answerer.generate_answer(query, retrieved)
    st.text_area("Raw Answer", raw_answer, height=220)

    # Critiqued refined answer
    st.markdown('<div class="sub-header">Refined Answer</div>', unsafe_allow_html=True)
    refined_answer = critiquer.critique(raw_answer)
    st.text_area("Refined Answer", refined_answer, height=260)


# ==========================================
# Follow-up Questions
# ==========================================
if followup_button:

    if not followup_query.strip():
        st.error("Follow-up question cannot be empty.")
        st.stop()

    st.markdown('<div class="section-header">Follow-up Answer</div>', unsafe_allow_html=True)

    retriever = RetrieverAgent()
    answerer = AnswerAgent()
    critiquer = CritiqueAgent()

    st.write("Retrieving relevant chunks...")
    retrieved_q = retriever.retrieve(followup_query)

    with st.expander("Retrieved Chunks for Follow-up"):
        for idx, m in enumerate(retrieved_q, start=1):
            st.markdown(f"**Chunk {idx} — Score {m['score']:.4f}**")
            st.write(f"Title: {m['metadata'].get('title', '')}")
            st.write(m["text"])
            st.markdown("---")

    raw = answerer.generate_answer(followup_query, retrieved_q)
    refined = critiquer.critique(raw)

    st.markdown('<div class="sub-header">Raw Answer</div>', unsafe_allow_html=True)
    st.text_area("Raw", raw, height=200)

    st.markdown('<div class="sub-header">Refined Answer</div>', unsafe_allow_html=True)
    st.text_area("Refined", refined, height=260)
