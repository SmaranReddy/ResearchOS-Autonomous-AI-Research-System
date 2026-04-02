import os
from groq import Groq
from dotenv import load_dotenv
from typing import List


class CritiqueAgent:
    """
    Refines the AnswerAgent output by checking context-grounding,
    removing off-topic content, and improving clarity.
    Does NOT regenerate from scratch.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GROQ_API_KEY in .env file")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        print("✅ CritiqueAgent ready.")

    def critique(self, answer: str, context: List[str], query: str = "") -> str:
        context_text = "\n\n".join(context)
        if len(context_text) > 8000:
            context_text = context_text[:8000]

        prompt = f"""You are a strict academic editor. Refine the answer below — do NOT rewrite it from scratch.

USER QUERY: {query}

Step 1 — Enforce query scope:
- Identify the key entities/concepts explicitly mentioned in the USER QUERY above
- Scan the answer for any concept NOT present in the query
- REMOVE any sentence or bullet that discusses a concept not in the query
- If an irrelevant concept must be mentioned, reduce it to at most 3 words with no explanation
- Example: query asks "Compare transformers with RNNs" → remove any CNN content entirely

Step 2 — Remove redundancy:
- If "Explanation" and "Key Points" repeat the same idea, keep it ONLY in Key Points as a short bullet
- Remove any bullet that duplicates information already in the Explanation
- Merge similar bullets into one concise bullet
- Shorten bullets to fragments, not full repeated sentences

Step 3 — Preserve integrity:
- Do NOT add any new information not already present in the answer
- Preserve the **Sources:** section exactly as-is
- If the answer is already clean and scoped — return it unchanged

Output format (preserve exactly):
**Explanation:** (1-3 clear sentences, focused only on query entities)

**Key Points:**
- (short fragment, on-topic only)

**Sources:**
- (unchanged)

---
SOURCE CONTEXT (for grounding check only):
{context_text}
---

ANSWER TO REFINE:
{answer}

Return ONLY the refined answer. No commentary, no preamble.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict academic reviewer who only refines answers, never fabricates."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            refined = response.choices[0].message.content.strip()

            if refined == answer.strip():
                print("[CRITIQUE] no changes needed")
            else:
                # detect removed concepts dynamically — any word that was in original but not in refined
                import re
                original_words = set(re.findall(r'\b[A-Z][A-Za-z0-9]+\b', answer))
                refined_words  = set(re.findall(r'\b[A-Z][A-Za-z0-9]+\b', refined))
                removed = original_words - refined_words
                if removed:
                    print(f"[CRITIQUE] removed irrelevant concepts: {', '.join(sorted(removed))}")
                else:
                    print("[CRITIQUE] changes applied")
            return refined

        except Exception as e:
            print(f"⚠️ CritiqueAgent failed: {e}")
            return answer  # fall back to original answer on failure
