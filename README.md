# ReSearch — Adaptive Intelligent RAG for Academic Research

> An end-to-end, production-deployed Retrieval-Augmented Generation system that searches live academic repositories, ingests PDFs on demand, and streams grounded, self-critiqued answers with multi-signal confidence scoring.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Key Features](#5-key-features)
6. [Performance & Optimization](#6-performance--optimization)
7. [Design Decisions](#7-design-decisions)
8. [Challenges & Solutions](#8-challenges--solutions)
9. [Evaluation](#9-evaluation)
10. [Deployment](#10-deployment)
11. [How to Run Locally](#11-how-to-run-locally)
12. [Future Improvements](#12-future-improvements)
13. [Resume Bullet Points](#13-resume-bullet-points)

---

## 1. Problem Statement

General-purpose LLMs answer research questions from static training data — they hallucinate citations, miss recent papers, and cannot ground claims in specific documents. Existing RAG demos index a fixed corpus upfront and never update it, so queries on niche or emerging topics always degrade to fabricated answers.

**ReSearch solves three concrete problems:**

1. **Stale knowledge** — the vector index is populated lazily and on demand from live academic sources (arXiv, Semantic Scholar, ACL Anthology, OpenReview), so the corpus stays current without manual curation.
2. **Ungrounded answers** — a three-signal composite confidence score and a type-aware self-critique retry loop detect and correct hallucinated or incomplete answers before they reach the user.
3. **Serverless deployment constraints** — free-tier infrastructure requires careful optimization of startup sequences, connection reuse, and request budgeting to stay within latency and rate-limit ceilings.

---

## 2. Solution Overview

ReSearch is a multi-phase, adaptive RAG pipeline built around a central insight: **retrieval quality should drive pipeline decisions, not be assumed.** At three checkpoints during each request, a 5-rule policy evaluates whether the existing Pinecone index is sufficient or whether fresh papers need to be ingested.

End-to-end flow:

1. **Query transformation** — the user query is rewritten into multiple semantically distinct variations (and pronoun-resolved for follow-up questions) to maximise retrieval recall.
2. **Parallel retrieval** — all query variations are embedded locally (ONNX, no API cost) and fired at Pinecone simultaneously; results are merged and deduplicated.
3. **Adaptive ingestion gate** — a 5-rule multi-signal policy decides whether to trigger ingestion (web search → PDF download → chunk → embed → index). Background ingestion is used when context is borderline, so the current request is not blocked.
4. **Batched LLM reranking** — all candidate documents are scored for semantic relevance in a single Groq LLM call.
5. **Confidence fast path** — if rerank quality is already high, the LLM confidence call is skipped entirely (saves 1–3 s on most queries).
6. **Answer generation** — the LLM synthesizes a grounded answer; a transparent fallback path fires when context is weak.
7. **Self-critique retry loop** — a lightweight scorer evaluates correctness, completeness, and grounding. A type-aware retry fires if the score is below threshold, using a retrieval strategy matched to the failure type.
8. **SSE streaming with post-stream refinement** — tokens stream token-by-token. Post-stream critique fires after the last token, replacing low-quality answers without adding to perceived latency.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Next.js Frontend (Vercel)                        │
│  SSE stream consumer · confidence badge · source chips · retry badge     │
│  debug panel · per-stage latency · post-stream "refined" notice          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ POST /api/stream  (Next.js route proxy)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend (Render)                           │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │            Startup Warmup  (@app.on_event "startup")              │  │
│  │  1. fastembed ONNX load  +  JIT compile via dummy inference       │  │
│  │  2. Pinecone singleton init  +  TCP/TLS pre-warm via dummy query  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  POST /stream ──► rate_limit (20 RPM/IP) ──► cache check (LRU+TTL)      │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Phase 1 — Parallel Query Transform + Probe                     │    │
│  │   ThreadPoolExecutor(2):                                         │    │
│  │     ├─ Groq: resolve pronouns + generate 2 variations            │    │
│  │     └─ Pinecone: probe with original query (hides transform lag) │    │
│  │   → [resolved_query, variation1, variation2]                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Phase 2a — Parallel Pinecone Retrieval                         │    │
│  │   embed queries sequentially (GIL-safe)                          │    │
│  │   fire all Pinecone calls in parallel (pure I/O, shared timeout) │    │
│  │   → deduplicate → 5-rule ingestion check (doc count + cosine)    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Phase 2b — Batched LLM Rerank                                  │    │
│  │   ALL docs scored in ONE Groq call  (O(1) not O(N))             │    │
│  │   → 5-rule ingestion check (adds rerank variance signal)         │    │
│  │   → if weak: background daemon ingestion (non-blocking)          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                          │ background                         │
│         │                          ▼                                    │
│         │               ┌─────────────────────┐                        │
│         │               │  Background Ingest  │                        │
│         │               │  Tavily → PDF DL →  │                        │
│         │               │  chunk → embed →    │                        │
│         │               │  Pinecone upsert     │                        │
│         │               └─────────────────────┘                        │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Phase 2c — Confidence Fast Path                                │    │
│  │   if rerank median ≥ 5.0 with ≥3 docs:                          │    │
│  │     confidence = 0.63 + median×0.027  (no LLM call, saves 1-3s) │    │
│  │   else: Groq context-sufficiency score (LLM call)               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Phase 3a — AsyncGroq Streaming Answer                          │    │
│  │   → SSE token events to client while generating                  │    │
│  │   → weak context → transparent fallback response                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Phase 3b — Self-Critique Score  (skipped if conf ≥ 0.60)      │    │
│  │   score ∈ [0,1]  type: incomplete / incorrect / not_grounded     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Phase 3c — Type-Aware Retry  (if critique_score < 0.7)        │    │
│  │   incomplete    → expand top_k + 2 (broader coverage)           │    │
│  │   incorrect     → tighten top_k = 3 (higher precision)          │    │
│  │   not_grounded  → filter to rerank_score ≥ 6.0                  │    │
│  │   accept only if new_score ≥ original_score + 0.05              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Composite Confidence + Cache Write                             │    │
│  │   0.25×retrieval + 0.35×rerank + 0.40×LLM  (adaptive weights)  │    │
│  │   → derive status → write to LRU cache (if success)             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
              │                                         │
    ┌─────────┴──────────┐                  ┌──────────┴─────────────┐
    │  Pinecone Serverless│                  │  Groq Cloud            │
    │  (AWS us-east-1)    │                  │  llama-3.1-8b-instant  │
    │  384-dim cosine     │                  │  query transform       │
    │  namespace=default  │                  │  rerank · answer       │
    └────────────────────┘                  │  critique_score        │
                                             └────────────────────────┘
```

---

## 4. Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14, React 18, TypeScript, Tailwind CSS, react-markdown + remark-gfm |
| **Backend** | Python 3.11, FastAPI, Uvicorn (2 workers in prod) |
| **LLM** | Groq Cloud — `llama-3.1-8b-instant` (query transform, rerank, answer, critique) |
| **Embeddings** | fastembed + `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local ONNX — no API cost, no network latency) |
| **Vector DB** | Pinecone Serverless (AWS us-east-1, cosine similarity, index `re-search`) |
| **Web Search** | Tavily API (arXiv, Semantic Scholar, ACL Anthology, OpenReview) |
| **PDF Parsing** | PyPDF2 + publisher-specific URL normalization (arXiv, Springer, IEEE, ACM) |
| **Containerization** | Docker multi-stage builds (dev + prod), docker-compose |
| **Deployment** | Render (backend), Vercel (frontend) |
| **Observability** | Structured JSONL request logger (timestamp, confidence, latency breakdown, cache hit, error) |

---

## 5. Key Features

### Adaptive 5-Rule Ingestion Decision Policy

The ingestion trigger is not a simple threshold check. `should_trigger_ingestion()` evaluates five rules in priority order, using only the signals available at the current pipeline phase (rules requiring unavailable signals are skipped automatically):

| Rule | Condition | Reason | Strength |
|---|---|---|---|
| 0. Entity coverage | Key query tokens absent from retrieved doc text | `low_relevance` | strong |
| 1. Low coverage | `n_docs < 3` | `low_docs` | strong/moderate |
| 2. Weak retrieval | `cosine_sim < 0.3` AND `rerank_norm < 0.4` | `low_relevance` | strong/moderate |
| 3. Rerank noise | `variance > 9.0` AND `rerank_norm < 0.5` | `low_relevance` | moderate |
| 4. LLM uncertainty | `llm_score < 0.3` AND `rerank_norm < 0.6` | `low_confidence` | strong/moderate |

The function is called at **three pipeline phases** (Phase 2a: post-retrieval, Phase 2b: post-rerank, Phase 2c: post-LLM-confidence), with progressively richer signal sets. This staged design means expensive LLM confidence calls are never made when simpler signals already detect a problem.

### Type-Aware Self-Critique Retry Loop

After generating an answer, a lightweight LLM scorer evaluates it on correctness, completeness, and grounding. If `critique_score < 0.7`, a **type-aware retry** fires with a strategy matched to the failure diagnosis:

```
incomplete    → expand top_k + 2  (more context for broader coverage)
incorrect     → tighten top_k = 3 (fewer, higher-precision docs)
not_grounded  → filter to rerank_score ≥ 6.0  (grounded context only)
```

The retry is accepted only if it improves the critique score by ≥ 0.05; otherwise the original answer is restored. Phase 3b critique is skipped entirely when `confidence ≥ 0.60`, saving 2–3 Groq API calls for the common case.

### Composite Adaptive Confidence Scoring

Three orthogonal signals combined with weights that adapt to signal quality at inference time:

```
composite = w_ret × retrieval_norm  +  w_rer × rerank_norm  +  w_llm × llm_score

Base weights:  retrieval=0.25   rerank=0.35   LLM=0.40

Adaptive adjustments:
  n_docs < 3                →  −0.10 retrieval  → +0.10 rerank
  rerank_std > 3.0          →  +0.05 rerank     → −0.05 LLM
  llm_score < 0.30          →  −0.10 LLM        → +0.05 retrieval, +0.05 rerank
```

The UI renders this as an animated confidence bar (Low / Moderate / High / Very High) with percentage.

### SSE Streaming with Post-Stream Critique

The `/stream` endpoint returns tokens as Server-Sent Events. Three event types:

- `token` — append to displayed content (real-time)
- `refine` — **replace** displayed content (fires after last token when post-stream critique detects low quality)
- `done` — finalize with confidence, sources, latency breakdown, decision trace

The post-stream critique runs **after all tokens are already on screen**, so it has zero impact on perceived latency for high-confidence answers. For low-confidence answers, it either replaces content with an honest "Low Confidence Notice" (for `not_grounded`) or calls `CritiqueAgent` to enforce query scope.

### Index-Version-Aware LRU Response Cache

```python
key = SHA-256(normalised_query + history_json + "||v" + index_version)
TTL = 10 min  (CACHE_TTL_SECONDS)
Max = 100 entries  (CACHE_MAX_ENTRIES)
```

The `index_version` counter increments on every Pinecone upsert. Pre-ingestion cached answers are never returned after new papers are indexed — old keys simply become unreachable and expire naturally. Only `success` responses are cached; `fallback`, `low_confidence`, and `error` are never stored.

### Three-Level Caching

| Level | Location | Size | Strategy |
|---|---|---|---|
| Response | `core/cache.py` | 100 entries | LRU + 10-min TTL + index-version invalidation |
| Retrieval | `retrieval/retriever.py` | 500 entries | FIFO, keyed by `query::top_k` |
| Embedding | `ingestion/embeddings.py` | 1000 entries | FIFO, batch-encodes only cache misses |

### Conversational Pronoun Resolution

Follow-up queries ("how is GAN different from **them**?") are rewritten into fully self-contained questions ("How is GAN different from BERT and GPT?") before embedding. Embedding pronouns without context produces semantically meaningless vectors — this step prevents retrieval degradation in multi-turn conversations. Chat history is used only for reference resolution, never as a factual source.

### Per-IP Sliding-Window Rate Limiter

20 req/min per IP (configurable). Implemented as a FastAPI `Depends` injection on top of an `AbstractRateLimiter` base class. Swapping to a Redis-backed implementation requires changing only the `get_rate_limiter()` factory body.

### Structured JSONL Request Logging

Every request appends one JSON line to `backend/logs/requests.jsonl`:

```json
{
  "timestamp": "2026-04-06T12:34:56.789Z",
  "query": "What is RAG?",
  "status": "success",
  "confidence": 0.823,
  "latency": { "retrieve_ms": 310, "rerank_ms": 150, "llm_ms": 620 },
  "sources_count": 3,
  "cached": false,
  "error": null
}
```

---

## 6. Performance & Optimization

### Startup Warmup

The service runs a `@app.on_event("startup")` async handler before accepting traffic. It pre-loads the fastembed ONNX model, triggers a dummy inference to force JIT kernel compilation, and runs a dummy Pinecone query to establish the TCP connection upfront. This front-loads expensive one-time setup costs so they don't fall on the first user request.

The Dockerfile also pre-downloads the ONNX model at image build time:

```dockerfile
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"
```

### Parallel Query Transform + Pinecone Probe

`run_pipeline_to_context` runs query transformation (Groq API — pure I/O) and the initial Pinecone probe concurrently in a `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=2) as pool:
    _transform_fut = pool.submit(QueryTransformer().transform, query, history)
    _probe_fut     = pool.submit(retriever.retrieve, query, top_k)
```

Wall time ≈ `max(transform, probe)` instead of `sum(transform, probe)`. **Typical saving: 300–800 ms.**

### GIL-Aware Parallel Pinecone Queries

`retrieve_many` separates two fundamentally different workloads:

1. **Embed all queries sequentially** — CPU-bound ONNX tensor ops hold the GIL; threading them would serialize anyway
2. **Query Pinecone for all in parallel** — pure network I/O, GIL released, truly concurrent; a shared wall-clock deadline caps total wait at `timeout_s` regardless of query count

For 3 query variations: wall time ≈ `1 × pinecone_latency` instead of `3 × pinecone_latency`.

### Batched LLM Reranking: O(1) LLM Calls

All candidate documents are scored in **one** Groq call using a comma-separated output format:

```
Query: <query>
Passages: [1] text... [2] text... [N] text...
Reply ONLY: 7,3,9  (one number per passage)
```

Compared to N sequential calls, this saves **2–7 s** for a typical 5–10 doc candidate set.

### Rerank-Based Confidence Fast Path

When rerank median ≥ 5.0 with ≥ 3 docs, the LLM context-sufficiency call in Phase 2c is skipped:

```python
if len(ranked_docs) >= 3 and rerank_median >= 5.0:
    confidence = min(0.63 + rerank_median * 0.027, 0.90)  # no LLM call
```

This formula was calibrated to match actual LLM scores within ±0.05 in the high-confidence range. **Saves 1–3 s on the majority of queries.**

### Singleton Pattern + Connection Pool Reuse

`RetrieverAgent`, `Reranker`, and the fastembed embedding model are module-level singletons. Before this optimization, `_step_rerank` created a fresh `Reranker()` on every pipeline call, discarding the `httpx` connection pool maintained by the Groq client and forcing a new TCP/TLS handshake each time. The singleton eliminates this overhead entirely.

### 15-Second Time Budget Guard

`MAX_TOTAL_TIME = 15.0 s` monitors wall time before Phases 3b, 3c, and 3d. If the pipeline is near the ceiling, critique and retry are skipped while still returning the best available answer. This prevents tail-latency runaway on slow Groq responses.

---

## 7. Design Decisions

### Decision 1: Local ONNX Embeddings (fastembed)

**What:** Embeddings generated locally via fastembed's ONNX runtime, not an embedding API.

**Why:** Zero network round-trip, zero API cost, no rate limit. `all-MiniLM-L6-v2` (384-dim) fits in ~100 MB RAM, runs in <100 ms per batch, and the ONNX runtime is pre-compiled at startup eliminating first-inference stall.

**Trade-off:** Larger models (e.g., OpenAI `text-embedding-3-large`) produce higher-quality embeddings at the cost of network latency and per-call pricing. The quality gap is negligible for academic text retrieval at this scale.

### Decision 2: Background Ingestion (Non-Blocking Daemon Thread)

**What:** When the adaptive policy detects weak context, ingestion runs in a daemon thread rather than blocking the current request.

**Why:** Ingestion takes 30–120 s (Tavily search + PDF download + PyPDF2 extraction + batched embedding + Pinecone upsert). Blocking for that duration is unacceptable UX. The current user gets the best answer possible from existing context; the next request for the same topic benefits from freshly indexed papers.

**Trade-off:** The current response may be less grounded than it would be post-ingestion. This is explicitly surfaced as `status = "low_confidence"` with the decision trace explaining why.

### Decision 3: Groq `llama-3.1-8b-instant` for All LLM Calls

**What:** One model and one API client for all LLM tasks — query transform, reranking, answer generation, critique scoring, critique refinement.

**Why:** At ~1000 tokens in 2–5 s at zero cost on Groq's free tier, the 8B model fits within the 15 s pipeline budget and 30 RPM rate limit. Using one model simplifies the client (one singleton, one connection pool) and eliminates model-switching overhead.

**Trade-off:** A 70B model would produce better answers but exceeds the latency budget on the free tier and costs money. The 8B model is sufficient when paired with good retrieval, batched reranking, and the self-critique loop.

### Decision 4: Index-Version Cache Invalidation

**What:** Cache key includes a process-level version counter incremented on every Pinecone upsert.

**Why:** Standard TTL caches would serve pre-ingestion answers after new papers are indexed. Including `index_version` in the SHA-256 key makes stale entries unreachable without an explicit eviction step.

**Trade-off:** Cache hit rates reset to zero after each ingestion. Acceptable because ingestion is rare (only triggered by the adaptive policy for weak context) and TTL expires old entries naturally.

### Decision 5: Pydantic Request/Response Models with Boundary Validation

**What:** All API inputs and outputs use typed Pydantic models with field-level validators.

**Why:** `query_not_empty` validator rejects blank queries at the HTTP boundary before any pipeline work runs. Response models provide a stable contract for the frontend and make latency/confidence fields self-documenting.

---

## 8. Challenges & Solutions

### Challenge 1: GIL-Constrained Parallel Retrieval

**Problem:** Early attempt parallelised all retrieval with `ThreadPoolExecutor` — but SentenceTransformer's `encode()` acquires the Python GIL during ONNX tensor operations. Multiple threads embedding simultaneously simply serialized, adding thread overhead with zero speedup.

**Diagnosis:** Measured per-query embedding time with and without parallelism using `time.monotonic()`. Times were statistically identical, confirming GIL serialization.

**Solution:** Separated the two workloads in `retrieve_many`: embed all queries **sequentially** (GIL-bound, unavoidable), then fire all Pinecone calls **in parallel** (pure network I/O, GIL released). Shared wall-clock deadline prevents compounding timeouts across futures.

**Result:** For 3 query variations, Pinecone wall time dropped from `3 × latency` to `1 × latency`.

### Challenge 2: Reranker Discarding the HTTP Connection Pool Every Request

**Problem:** `_step_rerank` in the executor instantiated `Reranker()` fresh on every pipeline call. This discarded the `httpx` connection pool maintained inside the Groq client object, forcing a new TCP/TLS handshake for every rerank call.

**Diagnosis:** Per-call rerank latency was inconsistently high (300–800 ms) despite `max_tokens=50`. Traced variance to connection establishment time by isolating first call vs. subsequent calls in a tight loop — first call was always ~400 ms slower.

**Solution:** Module-level `_reranker_instance` singleton via `get_reranker()`, mirroring the same pattern applied to `RetrieverAgent`.

**Result:** Rerank latency stabilized consistently in the 300–400 ms range.

### Challenge 3: LLM Confidence Call Adding Latency on High-Quality Queries

**Problem:** Phase 2c always called the LLM to score context sufficiency — even when the reranker had just confirmed strong relevance (docs scoring 7–9/10). This added a full Groq round-trip (~1–3 s) to every query, regardless of how obvious the context quality was.

**Solution:** Rerank-median fast path: if `median_rerank_score ≥ 5.0` with ≥ 3 docs, confidence is estimated from the median directly via `min(0.63 + median × 0.027, 0.90)`. This formula was calibrated empirically to match actual LLM scores within ±0.05 in the high-confidence range.

**Result:** LLM confidence call eliminated for the majority of queries. The Groq call still fires when context looks weak — the case where an accurate score matters most.

### Challenge 4: Retry Loop Permanently Blocked by Decision Strength

**Problem:** An early implementation gated the self-critique retry on `_decision_strength == "weak"` in addition to `critique_score < 0.7`. For well-indexed queries, `_decision_strength` was always `"strong"` — permanently blocking retry even when the critique score was low and the answer was genuinely incomplete.

**Diagnosis:** Added `[RETRY_DEBUG]` log lines exposing all gate variables on every pipeline call. Found cases where `critique_score=0.45`, `critique_type="incomplete"`, but `decision_strength="strong"` was silently blocking the retry.

**Root cause:** `_decision_strength` reflects retrieval uncertainty, not answer quality. These are orthogonal signals.

**Solution:** Removed `_decision_strength` from the retry gate entirely. The `critique_score` is the correct and sufficient signal for whether the answer needs improvement.

### Challenge 5: Groq API Rate Limit Exhaustion During Evaluation

**Problem:** The offline evaluator runs the full pipeline per test query — each query makes 3–4 Groq calls (query transform, rerank, answer, critique). With 10+ test queries, the 30 RPM free-tier limit is exhausted mid-run, causing failures.

**Solution:** The Phase 3b early-exit (`confidence ≥ 0.60` skips critique scoring) and the rerank-based confidence fast path together reduce the typical query from 4 Groq calls to 2–3. This keeps bulk evaluation within the rate limit without throttling logic.

---

## 9. Evaluation

The evaluation harness in `backend/evaluation/evaluator.py` scores answers on two complementary dimensions:

**1. Keyword Coverage (Baseline)**

Each test case in `test_queries.json` specifies expected technical keywords. The score is the fraction present in the answer.

**2. LLM Evaluation (Primary)**

`LLMEvaluator` calls Groq with a structured prompt returning JSON scores:
- **Relevance (1–10):** does the answer directly address the query?
- **Correctness (1–10):** is the content factually accurate and grounded, with no hallucinations?

**Output format:**

```
==============================
OVERALL SUMMARY
  Keyword score   : X/Y (Z%)
  Avg relevance   : N.N/10
  Avg correctness : N.N/10
==============================
```

The `batch-tests/` directory contains standalone scripts for testing individual components: Pinecone retrieval, embedding quality, Tavily search, and arXiv paper download.

---

## 10. Deployment

### Backend — Render

- Docker image from `backend/Dockerfile` multi-stage build (`base` → `prod`)
- Production command: `uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2`
- 2 Uvicorn workers: each process is stateless (Pinecone and Groq are external). Note: each worker independently loads the ONNX model, so RAM scales linearly with worker count.
- `GET /health` returns `{"status": "ok", "uptime_seconds": N, "booted_at": "..."}` — intentionally zero heavy logic, must respond in < 100 ms. Used by Render health checks and UptimeRobot keep-alive pings.

### Frontend — Vercel

- Next.js 14 deployed to Vercel (zero-config)
- Next.js API routes (`/app/api/stream/`, `/app/api/query/`) proxy requests to the Render backend URL
- Environment variable `BACKEND_URL` set to the Render service URL

### Keep-Alive

UptimeRobot is configured to ping `/health` every 5 minutes to reduce spin-down frequency on the free tier.

### Local Docker

```bash
# Development — live-mounted source, Uvicorn without --workers
docker compose up --build
# → Backend: http://localhost:8000
# → Frontend: http://localhost:3000  (NOT http://0.0.0.0:3000)

# Production — source baked into image, 2 workers
docker compose -f docker-compose.prod.yml up --build
```

---

## 11. How to Run Locally

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Groq API Key](https://console.groq.com) — free tier: 30 RPM
- [Pinecone API Key](https://app.pinecone.io) — free tier: 1 index
- [Tavily API Key](https://tavily.com) — free tier: 1,000 req/month

### 1. Configure environment

```bash
git clone <repo-url>
cd ResearchOS-Autonomous-AI-Research-System
cp .env.example .env
# Fill in GROQ_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY
```

### 2. Option A — Docker (recommended)

```bash
docker compose up --build
```

First run downloads the ONNX model (~90 MB) — allow 2–3 minutes.
Open: **http://localhost:3000**

### 3. Option B — Manual

**Backend:**
```bash
pip install -r requirements.txt
cd backend
uvicorn api.app:app --reload --port 8000
```

**Frontend (new terminal):**
```bash
cd frontend-next
npm install
npm run dev
```

### 4. Run the evaluator

```bash
cd backend
python evaluation/evaluator.py
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | LLM inference (all Groq calls) |
| `PINECONE_API_KEY` | Yes | — | Vector storage and retrieval |
| `TAVILY_API_KEY` | Yes | — | Academic paper web search |
| `CACHE_TTL_SECONDS` | No | `600` | Response cache TTL (seconds) |
| `CACHE_MAX_ENTRIES` | No | `100` | Response cache max entries |
| `RATE_LIMIT_RPM` | No | `20` | Per-IP request limit per minute |
| `RATE_LIMIT_WINDOW` | No | `60` | Rate limit window (seconds) |

---

## 12. Future Improvements

### Scale
- **Redis-backed cache and rate limiter** — `AbstractRateLimiter` and the cache factory are already in place; swap the implementation body for distributed state across workers
- **Pinecone namespace isolation** — namespace by topic domain for improved retrieval precision and per-topic cache invalidation
- **Persistent ingestion queue** — replace the in-memory `_active_ingestions` set with a Redis set to survive worker restarts

### Better Models
- **Larger embedding model** — swap `all-MiniLM-L6-v2` (384-dim) for `all-mpnet-base-v2` (768-dim); requires re-indexing Pinecone to the new dimension
- **Local cross-encoder reranking** — replace the LLM-based reranker with `ms-marco-MiniLM` for deterministic, faster, and free reranking
- **Groq `llama-3.1-70b-versatile`** for answer generation on a paid tier where latency budget allows

### Infrastructure
- **OpenTelemetry tracing** — structured spans for end-to-end latency visibility across Groq, Pinecone, and Tavily
- **Paper deduplication across restarts** — track indexed paper URLs to skip re-ingestion of already-seen papers

---

## 13. Resume Bullet Points

- **Designed and deployed a production-grade 8-phase adaptive RAG pipeline** (query transform, parallel retrieval, batched reranking, 3-checkpoint ingestion gating, confidence fast path, answer generation, self-critique scoring, type-aware retry) with multi-signal confidence scoring and real-time SSE token streaming

- **Implemented GIL-aware parallel Pinecone retrieval** by separating CPU-bound ONNX embedding (sequential) from network-bound Pinecone queries (parallel via `ThreadPoolExecutor` with shared wall-clock deadline), reducing multi-query retrieval wall time from `N × latency` to `1 × latency`

- **Cut LLM reranking cost from O(N × latency) to O(latency)** by redesigning the reranker to score all candidate documents in a single batched Groq call with comma-separated numeric output, saving 2–7 s per request compared to sequential per-doc scoring

- **Eliminated the LLM confidence API call for the majority of queries** by implementing a rerank-median fast path: when rerank median ≥ 5.0 across ≥ 3 docs, confidence is estimated from the rerank score directly via a calibrated formula, saving 1–3 s per request

- **Built a type-aware self-critique retry loop** classifying answer failures as `incomplete`, `incorrect`, or `not_grounded` and applying a matched retrieval strategy per type (expand top_k, tighten precision, filter to high-grounding docs), with acceptance gated on ≥ 0.05 improvement to prevent regressions

- **Engineered an index-version-aware LRU response cache** where SHA-256 cache keys include a process-level version counter that increments on every Pinecone upsert, guaranteeing post-ingestion responses are never served from stale pre-ingestion cache entries without explicit eviction logic

- **Shipped real-time SSE token streaming with post-stream refinement** — tokens delivered immediately via `AsyncGroq`; a post-stream critique fires after the last token and replaces the answer with a scope-enforced refinement or transparent fallback, making critique cost invisible to perceived latency for high-confidence responses

- **Implemented a 5-rule staged ingestion decision policy** evaluated at three pipeline checkpoints with progressively richer signals (doc count → rerank variance → LLM score), triggering non-blocking background ingestion for borderline context and synchronous ingestion only when context is demonstrably insufficient

- **Optimised Groq API usage to fit within the 30 RPM free-tier limit** across query transform, batched reranking, answer generation, and critique — by combining an early-exit confidence gate (skips critique for high-confidence answers) with the rerank-based confidence fast path (skips LLM confidence call), reducing typical query cost from 4 Groq calls to 2–3

- **Containerised and shipped a full-stack AI system to production** with Docker multi-stage builds (dev + prod targets), 2-worker Uvicorn in production, `/health` endpoint for Render health checks and keep-alive pings, FastAPI backend on Render, and Next.js frontend on Vercel
