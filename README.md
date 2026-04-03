# ReSearch — Adaptive RAG Research Assistant

A production-structured Retrieval-Augmented Generation system built around **relevance-aware ingestion**, **confidence-gated answering**, and a **multi-stage quality pipeline**. Designed to answer research queries strictly from retrieved context — with no hallucination and a deterministic fallback when context is insufficient.

---

## 🚀 Overview

Most RAG systems retrieve documents, stuff them into a prompt, and hope the LLM does the right thing. ReSearch takes a different approach:

- **Retrieval quality is measured**, not assumed
- **Ingestion is adaptive** — triggered by relevance, not document count
- **Answers are gated** by a confidence score before generation
- **Critique is conditional** — fallback responses bypass reformatting entirely

The result is a system that either returns a well-grounded, refined answer — or explicitly refuses to answer. No hallucination, no false confidence.

---

## 🧠 Key Innovations

**Relevance-aware ingestion trigger**
Ingestion is not triggered by counting documents. It is triggered when the LLM-rated quality of retrieved content falls below a threshold — measured using median rerank score and the proportion of low-quality documents. A count-based trigger would fire on empty indexes but miss the harder case: many retrieved documents that are entirely off-topic.

**LLM-rated reranking as a quality signal**
The reranker scores each document 0–10 via an LLM cross-encoder prompt. These scores feed both the top-k selection and the relevance guard — reusing a single LLM pass for two purposes.

**Confidence-scored answering**
Before generating an answer, the system asks the LLM to rate context sufficiency on a 0–1 scale. Below 0.5, no answer is generated. This catches cases where rerank scores looked acceptable but the context still cannot address the specific query.

**Controlled retry with a hard cap**
Ingestion can fire at most once per query — via a `state.ingestion_done` flag. This prevents runaway retries while still handling the case where the initial index has no relevant content.

**Critique bypass for fallbacks**
When `state.is_fallback` is True, the critique agent is skipped entirely. This prevents the critique LLM from reformatting a one-line refusal into a structured `**Explanation:** / **Key Points:**` block — a subtle but real failure mode.

**Multi-query retrieval with HyDE**
Each query generates five search variants: the original, a HyDE (Hypothetical Document Embedding) passage, and three angle-shifted expansions. All are used for retrieval; results are deduplicated before reranking.

---

## 🏗️ Architecture

```
Query
  │
  ▼
QueryTransformer          — HyDE + 3 query expansions (5 variants total)
  │
  ▼
RetrieverAgent            — Pinecone vector search across all variants, dedup
  │
  ▼
[Count Guard]             — if raw docs < 3 or avg score < 0.3 → ingest
  │
  ▼
Reranker                  — LLM scores each doc 0–10, returns top-k
  │
  ▼
[Relevance Guard]         — median < 4.0 OR majority low-quality → ingest
  │
  ▼
[Confidence Pre-check]    — if not yet ingested and confidence < 0.5 → ingest
  │
  ▼
AnswerAgent               — confidence gate → generate structured answer
  │
  ▼
CritiqueAgent             — scope enforcement + deduplication (skipped for fallbacks)
  │
  ▼
Final Answer + Confidence Score
```

The **Executor** drives this pipeline via a step registry — each step is a pure function `(State) → State`. The **Planner** defines step ordering. State is a single dataclass passed through every step.

---

## 🔄 Pipeline Flow

1. **Query Transform** — Generate 5 query variants (original + HyDE + 3 expansions)
2. **Retrieve** — Query Pinecone with all variants, merge and deduplicate results
3. **Count Guard** — Check raw doc count and cosine score average; ingest if weak
4. **Rerank** — LLM scores all retrieved docs, keeps top-k (default: 5)
5. **Relevance Guard** — Compute median rerank score and low-quality doc ratio; ingest if poor
6. **Confidence Pre-check** — LLM rates context sufficiency (0–1); ingest if < 0.5 and not yet ingested
7. **Answer** — Confidence gate: if < 0.5 → return fallback; else generate structured answer
8. **Critique** — Remove off-topic entities, deduplicate content, enforce query scope; skipped for fallbacks
9. **Return** — `{ answer, confidence, history }`

**Ingestion sub-pipeline** (triggered by guards):
`search_web → download → preprocess → chunk → embed → index → re-retrieve`

---

## ⚙️ Features

- **Multi-query retrieval**: HyDE + query expansion → broader semantic coverage per query
- **LLM reranker**: Cross-encoder-style scoring (0–10) per document, not cosine similarity alone
- **Three-layer quality gate**: count guard → relevance guard → confidence pre-check
- **Confidence scoring**: 0–1 float returned alongside every answer via the API
- **Controlled ingestion retry**: Hard cap of 1 ingestion per query via `state.ingestion_done`
- **Fallback safety**: Answers below confidence threshold never reach the LLM generation step
- **Critique agent**: General-purpose post-processing — entity-scope filtering, redundancy removal, citation preservation
- **Critique bypass**: `state.is_fallback` flag prevents critique from reformatting refusals
- **Deterministic post-filter**: Regex-based sentence filter removes off-topic entities after LLM critique
- **Caching**: Embedding and retrieval results cached to avoid redundant API calls
- **Strict grounding**: Answer prompt forbids outside knowledge; history block explicitly marked non-factual
- **Modular step registry**: Each pipeline step is independently testable and replaceable

---

## 🧪 Example Behavior

**Case 1 — Good context**
```
Query:    "Compare transformers with RNNs for sequence modeling"
Rerank:   [8.5, 7.9, 7.2, 6.8, 6.1]  median=7.2  → sufficient
Confidence: 0.84  → answer generated
Critique: removes any CNN content, deduplicates bullet points
Output:   structured answer with sources
```

**Case 2 — Weak initial index, successful ingestion**
```
Query:    "Explain diffusion models for protein structure prediction"
Rerank:   [3.1, 2.8, 2.4, 1.9, 1.2]  median=2.4  → low
→ ingestion triggered (search + download + embed + index)
→ re-retrieve + re-rerank: [8.1, 7.4, 6.9, 6.3, 5.8]  median=6.9  → sufficient
Confidence: 0.79  → answer generated
```

**Case 3 — Ingestion does not help**
```
Query:    "What is the thermodynamic entropy of black holes in LQG?"
Ingestion triggered → re-retrieved docs still off-topic
Confidence: 0.31  → below threshold
Output:   "The available sources do not contain sufficient information to answer this question reliably."
Critique: skipped (state.is_fallback = True)
```

---

## 📦 Installation & Setup

**Requirements**: Python 3.10+, Pinecone account, Groq API key, Tavily API key

```bash
git clone <repo-url>
cd re-search
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
TAVILY_API_KEY=your_tavily_key
```

---

## ▶️ Usage

**Backend API (FastAPI)**
```bash
cd backend
../.venv/Scripts/python -m uvicorn api.app:app --reload
```
API available at `http://127.0.0.1:8000`

**POST /query**
```json
{ "query": "Compare transformers with RNNs" }
```
```json
{
  "answer": "**Explanation:** ...\n\n**Key Points:**\n- ...\n\n**Sources:**\n- ...",
  "confidence": 0.81,
  "history": [...]
}
```

**Frontend UI**
Open `frontend/index.html` directly in a browser while the backend is running. No build step required.

**CLI**
```bash
cd backend
python main.py
```

---

## 🧩 Design Decisions

**Why confidence scoring instead of YES/NO sufficiency?**
A binary check loses resolution. A score of 0.48 vs 0.12 both fail the threshold, but the former may be worth retrying after ingestion while the latter likely won't improve. The float also surfaces in the API response, giving callers the option to display or act on it.

**Why median for the relevance guard?**
Average rerank score is skewed by outliers. A single highly relevant document in a set of irrelevant ones could mask a bad retrieval. Median is resistant to this. The additional `low_quality_count >= total // 2` condition catches the case where most documents are weak but none are extreme enough to drag the median below threshold alone.

**Why max-1 ingestion retry?**
Ingestion is expensive — it involves web search, downloading, preprocessing, embedding, and indexing. Allowing unbounded retries would make the system unusable for queries that genuinely have no indexed content. One retry is enough to handle a cold-start index; further retries are unlikely to help and add significant latency.

**Why skip critique for fallback responses?**
The critique agent uses an LLM prompt that expects structured content (`**Explanation:**`, `**Key Points:**`, `**Sources:**`). When given a single plain sentence, it reformats it into that structure — producing a fake "explanation" of a refusal. Skipping critique for fallbacks preserves the minimal, unambiguous message.

**Why a deterministic post-filter in the critique agent?**
LLMs don't reliably follow instructions to remove specific content. After the LLM critique pass, a regex-based sentence filter re-checks the answer against query entities. This catches off-topic content the LLM left in, without relying on prompt compliance.

---

## 📈 Future Improvements

- **Confidence calibration** — The 0–1 score is LLM-estimated; calibrate against a labeled set to make the 0.5 threshold more principled
- **Hybrid retrieval** — Combine dense (Pinecone) with sparse (BM25) retrieval; infrastructure is partially in place (`hybrid_search.py`)
- **Streaming responses** — Stream answer tokens to the frontend instead of blocking on full generation
- **Persistent cache** — Move embedding/retrieval cache from in-memory to disk or Redis for cross-session reuse
- **Evaluation harness** — Automate answer quality measurement using the existing `evaluation/` module
- **Deployment** — Containerize backend + frontend; add health check endpoint

---

## 👤 Author

**Smaran Reddy**
Built as a system-design-first exploration of production RAG patterns — focusing on reliability, modularity, and explicit failure handling over raw benchmark performance.

---

*Stack: Python · FastAPI · Pinecone · Groq (Llama 3.1) · Tavily · Streamlit (alt UI)*
