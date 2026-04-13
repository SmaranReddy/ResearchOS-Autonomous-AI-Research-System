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
from agents.answer_agent import AnswerAgent
from retrieval.retriever import get_retriever

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
async def _warmup_pinecone():
    """
    Fire a cheap Pinecone query on server startup so the first real user
    request does not pay the cold-start penalty (~3-8s on free-tier serverless).
    Also pre-loads the SentenceTransformer model into memory.
    """
    try:
        retriever = get_retriever()          # creates singleton + loads embedding model
        await asyncio.to_thread(retriever.retrieve, "warmup", top_k=1)
        print("[WARMUP] Pinecone + embedding model ready")
    except Exception as exc:
        print(f"[WARMUP] non-fatal warm-up error: {exc}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health — lightweight liveness probe used by Docker healthcheck
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


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
