import os
import re
import time
from typing import AsyncGenerator, Optional
from groq import Groq, AsyncGroq
from dotenv import load_dotenv

_PRIMARY_MODEL  = "llama-3.1-8b-instant"
_FALLBACK_MODEL = "llama3-70b-8192"   # larger model as backstop
_RETRY_DELAY    = 1.0                  # seconds between retries


class AnswerAgent:
    """
    Uses Groq's LLM to synthesize a coherent answer
    from the retrieved text context.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GROQ_API_KEY in .env file")

        self.client = Groq(api_key=api_key)
        self._api_key = api_key
        print("✅ AnswerAgent ready (Groq LLM).")

    CONFIDENCE_THRESHOLD = 0.5  # below this → context is treated as insufficient

    def _call_with_retry(self, messages: list, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Try primary model twice (with a 1s delay), then fall back to the larger model once.
        Raises on all failures.
        """
        attempts = [
            (_PRIMARY_MODEL, "primary attempt 1"),
            (_PRIMARY_MODEL, "primary attempt 2"),
            (_FALLBACK_MODEL, "fallback model"),
        ]
        last_exc: Exception | None = None
        for model, label in attempts:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if label != "primary attempt 1":
                    print(f"[LLM RETRY] succeeded on {label}")
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ LLM call failed ({label}): {e}")
                last_exc = e
                if model == _PRIMARY_MODEL:
                    time.sleep(_RETRY_DELAY)
        raise last_exc  # type: ignore[misc]

    def get_context_confidence(self, query: str, context_text: str) -> float:
        """
        LLM-based context confidence score.
        Returns a float in [0.0, 1.0] indicating how sufficient the context is.
        0.0 = no relevant information, 1.0 = fully sufficient.
        Fails closed (returns 0.0) on any error.
        """
        if not context_text.strip():
            return 0.0
        prompt = (
            f"Given the query and context, rate how sufficient the context is "
            f"to answer the query.\n"
            f"Return ONLY a number between 0 and 1.\n\n"
            f"0 = no relevant information\n"
            f"1 = fully sufficient information\n\n"
            f"Query: {query}\n\n"
            f"Context (excerpt):\n{context_text[:3000]}"
        )
        try:
            raw = self._call_with_retry(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            match = re.search(r"\d+(?:\.\d+)?", raw)
            if not match:
                return 0.0
            score = float(match.group())
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ Confidence check failed ({e}) — returning 0.0 (fail closed)")
            return 0.0

    def _extract_sources(self, context) -> list:
        """Extract structured source list from ranked docs."""
        sources = []
        if context and isinstance(context[0], dict):
            seen = set()
            for doc in context:
                meta = doc.get("metadata", {}) or {}
                title = meta.get("title", "")
                url = meta.get("url", "")
                if title and title not in seen:
                    seen.add(title)
                    sources.append({"title": title, "url": url})
        return sources

    def _build_context_text(self, context) -> tuple[str, list]:
        """
        Returns (context_text, sources_list).
        Accepts List[str] or List[Dict].
        """
        sources = self._extract_sources(context)
        if context and isinstance(context[0], dict):
            context = [doc.get("text", "") for doc in context]
        context_text = "\n\n".join(context)
        if len(context_text) > 15000:
            context_text = context_text[:15000]
        return context_text, sources

    def _build_prompt(self, query: str, context_text: str, chat_history: list) -> str:
        history_block = ""
        if chat_history:
            turns = chat_history[-2:]
            lines = []
            for turn in turns:
                q = turn.get("query", "")
                a = turn.get("answer", "")
                if q:
                    lines.append(f"Q: {q}")
                if a:
                    lines.append(f"A: {a[:300]}...")
            if lines:
                history_block = (
                    "Conversation History (use to resolve references like 'them', 'it', 'compare with' — "
                    "do NOT add new factual claims from history that are absent from the current context):\n"
                    + "\n".join(lines) + "\n\n"
                )

        return f"""You are a research assistant. Answer the question using ONLY the current context provided below.
Do NOT use any outside knowledge. Do NOT hallucinate or infer beyond what is stated.
Use the conversation history ONLY to resolve pronouns and references (e.g. "them", "it", "compare with") — do NOT treat past answers as new factual sources.

If the context does not contain enough information to answer, respond with exactly:
"I don't have enough information to answer this question."

---
{history_block}Current Context:
{context_text}
---

Current Question: {query}

Respond in this structure:
**Explanation:** (2-4 sentences directly answering the question from the context)

**Key Points:**
- (bullet point from context)
- (bullet point from context)
- (add more as needed)
"""

    def generate_answer(self, query: str, context, chat_history: list = [], state=None) -> str:
        """
        Synthesize an answer from ranked context docs.
        If state is provided:
          - reads state.confidence_cached to skip redundant LLM confidence call
          - writes state.confidence, state.is_fallback, state.sources
        Sources are stored on state.sources — NOT embedded in the returned answer text.
        """
        context_text, sources = self._build_context_text(context)

        if state is not None:
            state.sources = sources

        if not context_text.strip():
            if state is not None:
                state.is_fallback = True
            return "I don't have enough information to answer this question."

        # --- confidence gate ---
        # Reuse pre-computed confidence from Phase 2c if available (eliminates duplicate LLM call)
        if state is not None and state.confidence_cached:
            confidence = state.confidence
            print(f"[ANSWER] reusing cached confidence: {confidence:.2f}")
        else:
            confidence = self.get_context_confidence(query, context_text)
            print(f"[ANSWER CHECK] confidence: {confidence:.2f}")

        if state is not None:
            state.confidence = confidence

        if confidence < self.CONFIDENCE_THRESHOLD:
            if state is not None:
                state.is_fallback = True
            return "The available sources do not contain sufficient information to answer this question reliably."

        if state is not None:
            state.is_fallback = False

        prompt = self._build_prompt(query, context_text, chat_history)

        try:
            return self._call_with_retry(
                [
                    {
                        "role": "system",
                        "content": "You are a research assistant that answers only from provided context. Never fabricate information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )
        except Exception as e:
            print(f"⚠️ AnswerAgent failed ({e})")
            if state is not None:
                state.is_fallback = True
            return "An error occurred while generating the answer. Please try again."

    async def stream_answer(self, query: str, state) -> AsyncGenerator[str, None]:
        """
        Async generator that streams the answer token-by-token via Groq's streaming API.
        Computes confidence before streaming (one sync call wrapped in to_thread).
        Sets state.confidence, state.is_fallback, state.sources.
        Does NOT run CritiqueAgent — caller is responsible for post-processing if needed.
        """
        import asyncio

        context_text, sources = self._build_context_text(state.ranked_docs)
        state.sources = sources

        if not context_text.strip():
            state.is_fallback = True
            state.confidence = 0.0
            yield "I don't have enough information to answer this question."
            return

        # Confidence check (sync → thread)
        if state.confidence_cached:
            confidence = state.confidence
        else:
            confidence = await asyncio.to_thread(self.get_context_confidence, query, context_text)
        state.confidence = confidence
        print(f"[STREAM] confidence: {confidence:.2f}")

        if confidence < self.CONFIDENCE_THRESHOLD:
            state.is_fallback = True
            yield "The available sources do not contain sufficient information to answer this question reliably."
            return

        state.is_fallback = False
        prompt = self._build_prompt(query, context_text, state.chat_history)

        async_client = AsyncGroq(api_key=self._api_key)
        try:
            stream = await async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant that answers only from provided context. Never fabricate information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
                stream=True,
            )
            async for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as e:
            print(f"⚠️ stream_answer failed: {e}")
            yield "\n\n[Error: failed to stream response]"
