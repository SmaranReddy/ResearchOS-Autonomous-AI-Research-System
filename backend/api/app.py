import sys
import os
import json
import asyncio

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

app = FastAPI(title="re-search API")

_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

ResponseStatus = Literal["success", "low_confidence", "fallback", "error"]


class QueryRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None

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
        ),
        decision_trace=DecisionTrace(**(decision_trace or {})),
    )


# ---------------------------------------------------------------------------
# POST /query — synchronous, cached, with rate limiting
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
def query(request: QueryRequest):
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
        return cached

    # --- Pipeline ---
    try:
        answer, confidence, updated_history, sources, latency_ms, status, decision_trace = run_pipeline(
            request.query,
            chat_history=history,
        )
    except Exception as e:
        print(f"⚠️ Pipeline error: {e}")
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

    response = _build_response(answer, confidence, updated_history, sources, latency_ms, status, decision_trace)

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
        history = body.history or []

        # Phases 1–2: retrieval (blocking — run in thread pool)
        try:
            state = await asyncio.to_thread(
                run_pipeline_to_context,
                body.query,
                5,
                history,
            )
        except Exception as e:
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

        # Phase 3: stream LLM answer tokens
        async for token in agent.stream_answer(body.query, state):
            full_answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

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

        log_request(
            query=body.query,
            status=state.status,
            confidence=state.confidence,
            latency=state.latency_ms,
            sources_count=len(state.sources),
            cached=False,
        )

        yield f"data: {json.dumps({'type': 'done', 'confidence': state.confidence, 'status': state.status, 'sources': state.sources, 'history': state.chat_history, 'latency': state.latency_ms, 'decision_trace': state.decision_trace, 'critique_type': state._critique_type, 'critique_score': state._critique_score, 'retried': state._retried, 'decision_strength': state._decision_strength})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
