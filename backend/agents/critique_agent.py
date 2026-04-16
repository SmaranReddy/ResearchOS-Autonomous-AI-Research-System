import os
import re
from groq import Groq
from dotenv import load_dotenv
from typing import List


# ---------------------------------------------------------------------------
# Stop-word sets (module-level for reuse)
# ---------------------------------------------------------------------------

# Terms that appear capitalized in sentences but are NOT technical entities
_ANSWER_STOP: set = {
    'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'of', 'to', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
    'this', 'that', 'these', 'those', 'however', 'therefore', 'furthermore',
    'additionally', 'such', 'also', 'both', 'each', 'other', 'more', 'less',
    'first', 'second', 'third', 'finally', 'lastly', 'overall', 'unlike',
    'like', 'similar', 'different', 'while', 'whereas', 'when', 'where',
    'which', 'who', 'what', 'how', 'why', 'with', 'from', 'and', 'or', 'but',
    'not', 'no', 'use', 'used', 'using', 'based', 'since', 'if', 'then',
    'than', 'as', 'so', 'thus', 'hence', 'note', 'key', 'their', 'they',
    'them', 'its', 'it', 'he', 'she', 'we', 'our', 'your', 'my', 'one', 'two',
    'three', 'new', 'old', 'high', 'low', 'long', 'short', 'large', 'small',
    'same', 'type', 'types', 'task', 'tasks', 'step', 'steps', 'way', 'ways',
    'approach', 'method', 'methods', 'result', 'results', 'example', 'include',
    'includes', 'including', 'via', 'per', 'vs', 'and', 'also',
}

# Noise words to strip when extracting entities from the user query
_QUERY_STOP: set = {
    'compare', 'comparing', 'comparison', 'what', 'is', 'are', 'the', 'a', 'an',
    'how', 'does', 'do', 'with', 'vs', 'versus', 'between', 'and', 'or', 'of',
    'in', 'to', 'for', 'on', 'at', 'by', 'from', 'explain', 'describe', 'tell',
    'me', 'about', 'difference', 'differences', 'similarity', 'similarities',
    'which', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should',
    'use', 'used', 'using', 'give', 'show', 'list', 'summarize', 'define',
    'better', 'best', 'worse', 'worst', 'more', 'less', 'than', 'that', 'this',
    'overview', 'summary', 'brief', 'advantages', 'disadvantages', 'explain',
    'please', 'quickly', 'simply',
}


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
        # 12 s timeout — refinement call sends a long prompt but must not block pipeline
        self.client = Groq(api_key=api_key, timeout=12.0)
        self.model = "llama-3.1-8b-instant"
        print("✅ CritiqueAgent ready.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def critique(self, answer: str, context: List[str], query: str = "") -> str:
        context_text = "\n\n".join(context)
        if len(context_text) > 4000:
            context_text = context_text[:4000]  # reduced from 8000 — cuts prompt tokens by ~half

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
                max_tokens=600,  # reduced from 1000 — refinement doesn't need full token budget
            )
            refined = response.choices[0].message.content.strip()

            # --- Deterministic post-filter: enforce query scope ---
            if query:
                refined = self._post_filter(refined, query)

            if refined == answer.strip():
                print("[CRITIQUE] no changes needed")
            else:
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        General entity extraction from the query.

        Returns a list of lowercased meaningful tokens that represent the
        focus of the query (e.g. ['transformers', 'rnns'] for a comparison
        query). No domain-specific hardcoding.
        """
        tokens = re.findall(r'\b\w+\b', query.lower())
        return [t for t in tokens if t not in _QUERY_STOP and len(t) > 2]

    def _post_filter(self, answer: str, query: str) -> str:
        """
        Deterministic post-filter that removes lines/bullets whose ONLY
        named entities are unrelated to the query.

        Algorithm (general — no hardcoded terms):
        1. Extract focus entities from query.
        2. Split answer into main body + Sources (Sources preserved as-is).
        3. For each line in the main body:
           a. Always keep empty lines and section headers.
           b. Extract named technical terms (acronyms + PascalCase, ≥3 chars,
              not in the common-word stop list).
           c. If no named terms → generic statement → keep.
           d. If at least one named term overlaps (substring match) with a
              query entity → keep.
           e. Otherwise → remove (line is exclusively about off-topic entities).
        4. Log if anything was removed.
        """
        query_entities = self._extract_query_entities(query)
        if not query_entities:
            return answer  # cannot filter without a focus

        # Preserve Sources section intact
        sources_split = re.split(r'(\*\*Sources:\*\*.*)', answer, flags=re.DOTALL | re.IGNORECASE)
        main_content   = sources_split[0]
        sources_section = sources_split[1] if len(sources_split) > 1 else ''

        lines = main_content.split('\n')
        filtered_lines: List[str] = []
        removed_count = 0

        for line in lines:
            stripped = line.strip()

            # Keep empty lines and section headers (**...**)
            if not stripped or re.match(r'^\*\*.+\*\*', stripped):
                filtered_lines.append(line)
                continue

            # Extract named technical terms from the line:
            #   - Acronyms:   2+ uppercase letters  (RNN, CNN, GPT)
            #   - PascalCase: capital + 2+ lowercase (Transformer, Recurrent)
            raw_terms = re.findall(r'\b[A-Z]{2,}\b|\b[A-Z][a-z]{2,}(?:[A-Z][a-z]*)?\b', line)
            named = [t.lower() for t in raw_terms if t.lower() not in _ANSWER_STOP]

            if not named:
                # No named entities → generic statement, always keep
                filtered_lines.append(line)
                continue

            # Keep if ANY named term overlaps with ANY query entity
            def _overlaps(nl: str, qe: str) -> bool:
                return qe in nl or nl in qe

            relevant = any(
                any(_overlaps(nl, qe) for qe in query_entities)
                for nl in named
            )

            if relevant:
                filtered_lines.append(line)
            else:
                removed_count += 1

        result = '\n'.join(filtered_lines).rstrip()
        if sources_section:
            result += '\n' + sources_section

        if removed_count > 0:
            print(f"[CRITIQUE] removed irrelevant content based on query scope ({removed_count} line(s) filtered)")

        return result


# ---------------------------------------------------------------------------
# Module-level singleton — avoids recreating the Groq httpx connection pool
# on every critique call.
# ---------------------------------------------------------------------------

_critique_agent_instance: "CritiqueAgent | None" = None


def get_critique_agent() -> "CritiqueAgent":
    global _critique_agent_instance
    if _critique_agent_instance is None:
        _critique_agent_instance = CritiqueAgent()
    return _critique_agent_instance
