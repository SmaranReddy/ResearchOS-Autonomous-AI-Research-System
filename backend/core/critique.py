"""
Answer self-critique scorer.

Distinct from agents/critique_agent.py (which *refines* answer text).
This module *scores* an already-generated answer on three dimensions:
correctness, completeness, and grounding in context.

Returns a float score ∈ [0, 1] and a short reason string.
Used to decide whether a retry with expanded context is worthwhile.

Cost: 1 Groq LLM call per invocation (small model, max_tokens=80).
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Tuple

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    from core.llm_counter import record as _llm_record
except ImportError:
    def _llm_record(caller, model, elapsed_ms): pass

_MODEL = "llama-3.1-8b-instant"

_VALID_TYPES = {"incomplete", "incorrect", "not_grounded", "good"}

_PROMPT_TEMPLATE = """\
Evaluate how well the answer addresses the query given the provided context.

Score from 0 to 1 based on:
- correctness: is the answer factually accurate relative to the context?
- completeness: does it cover the key points needed to answer the query?
- grounding: is every claim traceable to the context (no hallucination)?

Choose the single most accurate critique type:
- "incomplete"    : answer is missing key information needed to fully address the query
- "incorrect"     : answer contains factual errors relative to the context
- "not_grounded"  : answer makes claims not supported by the provided context
- "good"          : answer is correct, complete, and grounded

Return EXACTLY three lines, nothing else:
score: <float between 0 and 1>
type: <one of: incomplete, incorrect, not_grounded, good>
reason: <one sentence explaining the score>

Query: {query}

Context (excerpt):
{context_excerpt}

Answer:
{answer}
"""


def critique_answer(
    query: str,
    answer: str,
    context: List[str],
) -> Tuple[float, str, str]:
    """
    Score the answer against the query and context.

    Parameters
    ----------
    query   : original user query
    answer  : generated answer to evaluate
    context : list of context text strings (from ranked_docs)

    Returns
    -------
    (score: float, reason: str, critique_type: str)
      score         ∈ [0.0, 1.0]
      reason        — one-sentence explanation
      critique_type ∈ {"incomplete", "incorrect", "not_grounded", "good"}
                      defaults to "good" on parse failure (fails open)
    """
    if not answer.strip() or not context:
        return 0.0, "empty answer or context", "incomplete"

    context_excerpt = "\n\n".join(context)[:4000]   # cheap truncation, no extra call

    prompt = _PROMPT_TEMPLATE.format(
        query=query,
        context_excerpt=context_excerpt,
        answer=answer[:2000],
    )

    try:
        api_key = os.getenv("GROQ_API_KEY")
        # 8 s timeout — scoring uses max_tokens=100, completes in <2 s normally
        client = Groq(api_key=api_key, timeout=8.0)
        _t0 = time.monotonic()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        _elapsed = int((time.monotonic() - _t0) * 1000)
        _llm_record("critique_answer (scorer)", _MODEL, _elapsed)
        print(f"[critique] LLM scoring took {_elapsed}ms")
        raw = response.choices[0].message.content.strip()

        score_match  = re.search(r"score\s*:\s*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)
        type_match   = re.search(r"type\s*:\s*(\w+)",               raw, re.IGNORECASE)
        reason_match = re.search(r"reason\s*:\s*(.+)",              raw, re.IGNORECASE)

        score  = float(score_match.group(1))  if score_match  else 0.5
        reason = reason_match.group(1).strip() if reason_match else raw[:120]

        raw_type     = type_match.group(1).lower().strip() if type_match else ""
        critique_type = raw_type if raw_type in _VALID_TYPES else "good"

        score = max(0.0, min(1.0, score))
        print(f"[CRITIQUE_SCORE] score={score:.2f}  type={critique_type}  reason={reason}")
        return score, reason, critique_type

    except Exception as exc:
        print(f"[CRITIQUE_SCORE] failed ({exc}) — defaulting to 0.5/good (fail open)")
        return 0.5, "scoring error", "good"
