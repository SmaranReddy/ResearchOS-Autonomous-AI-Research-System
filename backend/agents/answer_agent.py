# ==========================================
# backend/agents/answer_agent.py
# ==========================================

import os
from groq import Groq
from dotenv import load_dotenv


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
        print("✅ AnswerAgent ready (Groq LLM).")

    CONFIDENCE_THRESHOLD = 0.5  # below this → context is treated as insufficient

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
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip()
            # extract first float-like token from response
            import re
            match = re.search(r"\d+(?:\.\d+)?", raw)
            if not match:
                return 0.0
            score = float(match.group())
            return max(0.0, min(1.0, score))  # clamp to [0, 1]
        except Exception as e:
            print(f"⚠️ Confidence check failed ({e}) — returning 0.0 (fail closed)")
            return 0.0

    def generate_answer(self, query: str, context, chat_history: list = [], state=None) -> str:
        """
        Combine retrieved chunks into a final answer using Groq.
        Accepts either List[str] or List[Dict] (ranked_docs with 'text' key).
        chat_history: last N interactions as [{"query": ..., "answer": ...}]
        """
        # extract unique paper titles for citations before stripping to text
        citations = []
        if context and isinstance(context[0], dict):
            seen = set()
            for doc in context:
                title = doc.get("metadata", {}).get("title", "")
                if title and title not in seen:
                    seen.add(title)
                    citations.append(title)
            context = [doc.get("text", "") for doc in context]
        context_text = "\n\n".join(context)
        if len(context_text) > 15000:  # keep prompt size manageable
            context_text = context_text[:15000]

        if not context_text.strip():
            if state is not None:
                state.is_fallback = True
            return "I don't have enough information to answer this question."

        # --- confidence gate: no answer generated below threshold ---
        confidence = self.get_context_confidence(query, context_text)
        print(f"[ANSWER CHECK] confidence: {confidence:.2f}")
        if confidence < self.CONFIDENCE_THRESHOLD:
            if state is not None:
                state.is_fallback = True
            return "The available sources do not contain sufficient information to answer this question reliably."

        if state is not None:
            state.is_fallback = False

        # build conversation history block (last 2 turns)
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
                history_block = "Conversation History (for intent only — do NOT use as facts):\n" + "\n".join(lines) + "\n\n"

        prompt = f"""You are a research assistant. Answer the question using ONLY the current context provided below.
Do NOT use any outside knowledge. Do NOT hallucinate or infer beyond what is stated.
The conversation history is provided only to understand the user's intent — do NOT treat it as factual source.

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

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a research assistant that answers only from provided context. Never fabricate information."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            answer = response.choices[0].message.content.strip()
            if citations:
                citation_block = "\n\n**Sources:**\n" + "\n".join(f"- {t}" for t in citations)
                answer += citation_block
            return answer

        except Exception as e:
            print(f"⚠️ AnswerAgent failed ({e})")
            return ""

    def generate_overview(self, query, context=None):
        """Academic overview mode."""
        prompt = f"""
Write a full academic overview on the topic **{query}**, including:

1. Title  
2. Abstract  
3. Historical Evolution  
4. Architecture / Core Mechanism  
5. Applications  
6. Advantages  
7. Challenges and Limitations  
8. Future Research Directions  
9. Conclusion  
10. Keywords

Tone: formal, factual, and academic. Avoid citations.
        """
        for model in self.models:
            answer = self._call_model(prompt, model)
            if answer:
                print(f"✅ Academic overview generated using {model}")
                return answer
        return "❌ Failed to generate overview."

    def generate_precise_answer(self, question, context_summary=""):
        """Short, factual answers for user Q&A."""
        prompt = f"""
Answer precisely and concisely.

Context (if relevant):
{context_summary}

Question:
{question}

Guidelines:
- Maximum 5 sentences.
- Be direct, factual, and simple.
- Avoid academic tone or repetition.
        """
        for model in self.models:
            answer = self._call_model(prompt, model)
            if answer:
                print(f"✅ Precise Q&A generated using {model}")
                return answer
        return "❌ Could not generate response."
