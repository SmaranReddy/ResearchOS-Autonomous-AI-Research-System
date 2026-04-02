import sys
import os

# Ensure backend/ is on the path so imports work regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from core.executor import run_pipeline

app = FastAPI(title="re-search API")


class QueryRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None


class QueryResponse(BaseModel):
    answer: str
    history: List[dict]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    answer, updated_history = run_pipeline(
        request.query,
        chat_history=request.history or [],
    )
    return QueryResponse(answer=answer, history=updated_history)
