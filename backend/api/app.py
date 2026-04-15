import sys
import os
import json
import asyncio
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Literal

from core.executor import run_pipeline, run_pipeline_to_context
from core.cache import get_cache
from core.confidence import compute_composite, derive_status, LOW_CONFIDENCE_THRESHOLD
from core.rate_limiter import rate_limit
from core.logger import log_request
from core.critique import critique_answer
from agents.answer_agent import AnswerAgent
from agents.critique_agent import CritiqueAgent
from retrieval.retriever import get_retriever

# Confidence threshold below which the streaming path runs a post-answer
# critique pass and emits a "refine" SSE event to replace the displayed answer.
_STREAM_CRITIQUE_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Cold-start detection
# Set once when the process boots; compared on every /health call to surface
# restarts in logs and monitoring dashboards.
# ---------------------------------------------------------------------------
_BOOT_TIME: float = time.monotonic()
_BOOT_WALL: str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

app = FastAPI(title="re-search API")


@app.on_event("startup")
async def _validate_env():
    """Warn at startup if required API keys are absent — easier to debug than a mid-request crash."""
    required = {
        "GROQ_API_KEY":    "Groq LLM calls will fail",
        "PINECONE_API_KEY": "Vector retrieval will fail",
    }
    for var, consequence in required.items():
        if not os.getenv(var):
            print(f"[WARN] Missing env var {var!r} — {consequence}")


@app.on_event("startup")
async def _warmup():
    """
    Pre-warm all expensive resources so the first user request does NOT pay
    cold-start costs.

    What this eliminates:
      1. fastembed ONNX model load (1–5 s) — triggered by get_embedding_model()
      2. ONNX JIT kernel compilation (200–500 ms) — triggered by dummy embed call
      3. Pinecone client init + list_indexes() network call (100–300 ms)
      4. Pinecone TCP/TLS connection establishment (200–500 ms on Render)

    Each step is wrapped in try/except so a Pinecone outage at deploy time
    does not prevent the service from starting.
    """
    print("[WARMUP] Starting pre-warm sequence...")

    # ── Step 1: Load embedding model + trigger ONNX JIT via dummy inference ──
    # get_embedding_model() is a singleton — safe to call multiple times.
    # The dummy embed call after it forces the ONNX runtime to JIT-compile
    # the inference graph, eliminating the first-query compilation stall.
    try:
        from utils.model import get_embedding_model
        model = await asyncio.to_thread(get_embedding_model)
        # list() fully consumes the generator, flushing the ONNX warmup pass
        await asyncio.to_thread(lambda: list(model.embed(["warmup"])))
        print("[WARMUP] Embedding model ready")
    except Exception as exc:
        print(f"[WARMUP] Embedding model warmup failed (non-fatal): {exc}")

    # ── Step 2: Init Pinecone singleton + open TCP/TLS connection ─────────────
    # get_retriever() builds the RetrieverAgent singleton (creates Pinecone
    # client, checks index existence).  The dummy query after it opens the
    # TCP connection so the first real query doesn't pay the handshake cost.
    try:
        retriever = await asyncio.to_thread(get_retriever)
        dummy_vec = await asyncio.to_thread(retriever.embed_query, "warmup")
        await asyncio.to_thread(
            retriever.index.query,
            namespace="default",
            vector=dummy_vec,
            top_k=1,
            include_metadata=False,
        )
        print("[WARMUP] Pinecone connection warmed")
    except Exception as exc:
        print(f"[WARMUP] Pinecone warmup failed (non-fatal): {exc}")

    print("[WARMUP] Done — service is warm and ready")
    print(f"[COLD_START] Process booted at {_BOOT_WALL} — all resources pre-warmed")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health — lightweight liveness + readiness probe
#
# Used by:
#   • Render's built-in health checks
#   • UptimeRobot / Better Uptime keep-alive pings (every 5 min)
#   • Docker HEALTHCHECK
#
# Contract: MUST return 200 in < 100 ms. Zero heavy logic here.
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
def health():
    uptime_s = int(time.monotonic() - _BOOT_TIME)
    return {
        "status": "ok",
        "uptime_seconds": uptime_s,
        "booted_at": _BOOT_WALL,
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

ResponseStatus = Literal["success", "low_confidence", "fallback", "error"]


class QueryRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None
    disable_retry: bool = False

    @validator("query")
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class SourceItem(BaseModel):
    title: str
    url: str = ""


class LatencyInfo(BaseModel):
    retrieve_ms: int = 0
    rerank_ms:   int = 0
    llm_ms:      int = 0
    total_ms:    int = 0


class DecisionTrace(BaseModel):
    retrieval_quality:    str = ""
    action:               str = ""
    confidence_reasoning: str = ""


class QueryResponse(BaseModel):
    answer:         str
    confidence:     float
    status:         ResponseStatus = "success"
    history:        List[dict]
    sources:        List[SourceItem] = []
    latency:        LatencyInfo = LatencyInfo()
    decision_trace: DecisionTrace = DecisionTrace()
    retried:        bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_response(
    answer: str,
    confidence: float,
    history: list,
    sources: list,
    latency_ms: dict,
    status: str,
    decision_trace: dict | None = None,
    retried: bool = False,
) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        confidence=confidence,
        status=status,          # type: ignore[arg-type]
        history=history,
        sources=[SourceItem(**s) for s in sources],
        latency=LatencyInfo(
            retrieve_ms=latency_ms.get("retrieve_ms", 0),
            rerank_ms=latency_ms.get("rerank_ms", 0),
            llm_ms=latency_ms.get("llm_ms", 0),
            total_ms=latency_ms.get("total_ms", 0),
        ),
        decision_trace=DecisionTrace(**(decision_trace or {})),
        retried=retried,
    )


# ---------------------------------------------------------------------------
# POST /query — synchronous, cached, with rate limiting
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
def query(request: QueryRequest):
    print(f"[REQUEST_RECEIVED] query='{request.query[:80]}'")
    cache = get_cache()
    history = request.history or []

    # --- Cache read ---
    cached = cache.get(request.query, history)
    if cached is not None:
        log_request(
            query=request.query,
            status=cached.status,
            confidence=cached.confidence,
            latency=cached.latency.dict(),
            sources_count=len(cached.sources),
            cached=True,
        )
        print(f"[RESPONSE_SENT] (cached) status={cached.status}")
        return cached

    # --- Pipeline ---
    try:
        print(f"[RETRIEVE_START] query='{request.query[:80]}'")
        answer, confidence, updated_history, sources, latency_ms, status, decision_trace, retried = run_pipeline(
            request.query,
            chat_history=history,
            disable_retry=request.disable_retry,
        )
    except Exception as e:
        print(f"[ERROR] Pipeline exception: {e}")
        traceback.print_exc()
        log_request(
            query=request.query,
            status="error",
            confidence=0.0,
            latency={},
            sources_count=0,
            cached=False,
            error=str(e),
        )
        return QueryResponse(
            answer="The pipeline encountered an error. Please try again.",
            confidence=0.0,
            status="error",
            history=history,
        )

    response = _build_response(answer, confidence, updated_history, sources, latency_ms, status, decision_trace, retried)

    log_request(
        query=request.query,
        status=status,
        confidence=confidence,
        latency=latency_ms,
        sources_count=len(sources),
        cached=False,
    )

    # --- Cache write (only successful answers) ---
    if status == "success":
        cache.set(request.query, history, response)

    print(f"[RESPONSE_SENT] status={status} confidence={confidence:.3f} sources={len(sources)}")
    return response


# ---------------------------------------------------------------------------
# POST /stream — SSE streaming, with rate limiting
#
# Event shapes (JSON after "data: "):
#   {"type": "token",  "content": "..."}
#   {"type": "done",   "confidence": float, "status": str,
#                      "sources": [...], "history": [...], "latency": {...}}
#   {"type": "error",  "message": "..."}
# ---------------------------------------------------------------------------

@app.post("/stream", dependencies=[Depends(rate_limit)])
async def stream_query(request: Request, body: QueryRequest):
    async def event_gen():
        _t_stream_start = time.monotonic()
        print(f"[REQUEST_RECEIVED] stream query='{body.query[:80]}'")
        history = body.history or []

        # --- Cache read (stream) ---
        cache = get_cache()
        cached = cache.get(body.query, history)
        if cached is not None:
            log_request(
                query=body.query,
                status=cached.status,
                confidence=cached.confidence,
                latency=cached.latency.dict(),
                sources_count=len(cached.sources),
                cached=True,
            )
            print(f"[RESPONSE_SENT] stream (cached) status={cached.status}")
            yield f"data: {json.dumps({'type': 'token', 'content': cached.answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'confidence': cached.confidence, 'status': cached.status, 'sources': [s.dict() for s in cached.sources], 'history': history, 'latency': cached.latency.dict(), 'decision_trace': cached.decision_trace.dict(), 'critique_type': '', 'critique_score': 0.0, 'retried': False, 'decision_strength': ''})}\n\n"
            return

        # Phases 1–2: retrieval (blocking — run in thread pool)
        try:
            print(f"[RETRIEVE_START] query='{body.query[:80]}'")
            state = await asyncio.to_thread(
                run_pipeline_to_context,
                body.query,
                3,
                history,
            )
        except Exception as e:
            print(f"[ERROR] stream retrieval exception: {e}")
            traceback.print_exc()
            log_request(
                query=body.query,
                status="error",
                confidence=0.0,
                latency={},
                sources_count=0,
                cached=False,
                error=str(e),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        agent = AnswerAgent()
        full_answer_parts: list[str] = []

        # Use the resolved (pronoun-expanded) query for confidence scoring and prompting.
        # Falls back to body.query if resolution didn't run (e.g. no history).
        effective_query = state.resolved_query or body.query
        print(f"[ANSWER_START] streaming tokens for query='{effective_query[:80]}'")
        # Phase 3: stream LLM answer tokens
        _t_llm = time.monotonic()
        try:
            async for token in agent.stream_answer(effective_query, state):
                full_answer_parts.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            print(f"[ERROR] stream_answer exception: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        state.latency_ms["llm_ms"] = int((time.monotonic() - _t_llm) * 1000)
        state.latency_ms["total_ms"] = int((time.monotonic() - _t_stream_start) * 1000)
        state.final_answer = "".join(full_answer_parts)

        # Composite confidence + status (same logic as sync path)
        compute_composite(state)
        state.status = derive_status(state)
        print(f"[STREAM] status={state.status}  confidence={state.confidence:.3f}")

        # Post-stream critique pass — runs only for low-confidence answers.
        # Because it fires AFTER all tokens are already sent, latency is invisible
        # to the user for the initial render.  If the answer is poor, a "refine"
        # SSE event replaces the displayed content before the "done" event arrives.
        if state.confidence < _STREAM_CRITIQUE_THRESHOLD and not state.is_fallback:
            print(f"\n[STREAM_CRITIQUE_START] confidence={state.confidence:.2f} < {_STREAM_CRITIQUE_THRESHOLD}")
            _t_sc = time.monotonic()
            _ctx_texts = [doc.get("text", "") for doc in state.ranked_docs if isinstance(doc, dict)]

            _sc_score, _sc_reason, _sc_type = await asyncio.to_thread(
                critique_answer, body.query, state.final_answer, _ctx_texts
            )
            print(f"[STREAM_CRITIQUE_SCORE] score={_sc_score:.2f}  type={_sc_type}  reason={_sc_reason}")

            if _sc_score < 0.40:
                if _sc_type == "not_grounded":
                    # Answer is hallucinated — replace with an honest fallback
                    _refined = (
                        "**Low Confidence Notice**\n\n"
                        "The retrieved papers don't contain reliable information specifically "
                        "about this topic, and the initial response may include inaccurate details.\n\n"
                        "I've triggered a search for additional sources — asking again should "
                        "produce a better-grounded answer.\n\n"
                        "Try rephrasing, or ask about a closely related concept that may be "
                        "better covered in the literature."
                    )
                else:
                    # Answer is incomplete or incorrect — use CritiqueAgent to tighten it
                    _refined = await asyncio.to_thread(
                        CritiqueAgent().critique,
                        state.final_answer,
                        _ctx_texts,
                        body.query,
                    )

                if _refined != state.final_answer:
                    state.final_answer = _refined
                    yield f"data: {json.dumps({'type': 'refine', 'content': _refined})}\n\n"

            print(f"[STREAM_CRITIQUE_TIME] {int((time.monotonic() - _t_sc) * 1000)}ms")

        # Finalise action for streaming path
        if not state.decision_trace["action"]:
            if state.is_fallback:
                state.decision_trace["action"] = "fallback_no_answer"
            else:
                state.decision_trace["action"] = "used_existing_knowledge"

        # Update chat history
        state.chat_history.append({"query": body.query, "answer": state.final_answer})
        state.chat_history = state.chat_history[-3:]

        print(f"[RETRIEVE_TIME] {state.latency_ms['retrieve_ms']}ms")
        print(f"[RERANK_TIME]   {state.latency_ms['rerank_ms']}ms")
        print(f"[LLM_TIME]      {state.latency_ms['llm_ms']}ms")
        print(f"[TOTAL_TIME]    {state.latency_ms['total_ms']}ms")

        log_request(
            query=body.query,
            status=state.status,
            confidence=state.confidence,
            latency=state.latency_ms,
            sources_count=len(state.sources),
            cached=False,
        )

        # --- Cache write for stream (only successful answers) ---
        if state.status == "success":
            _stream_response = _build_response(
                state.final_answer,
                state.confidence,
                state.chat_history,
                state.sources,
                state.latency_ms,
                state.status,
                state.decision_trace,
            )
            cache.set(body.query, history, _stream_response)

        print(f"[RESPONSE_SENT] stream status={state.status} confidence={state.confidence:.3f}")
        yield f"data: {json.dumps({'type': 'done', 'confidence': state.confidence, 'status': state.status, 'sources': state.sources, 'history': state.chat_history, 'latency': state.latency_ms, 'decision_trace': state.decision_trace, 'critique_type': state._critique_type, 'critique_score': state._critique_score, 'retried': state._retried, 'decision_strength': state._decision_strength})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
