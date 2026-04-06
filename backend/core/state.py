from typing import List, Dict, Optional


class State:
    def __init__(self, user_query: str = "", num_papers: int = 5):
        # --- core inputs ---
        self.user_query: str = user_query
        self.num_papers: int = num_papers

        # --- ingestion ---
        self.papers: List[Dict] = []
        self.cleaned_texts: List[str] = []
        self.token_counts: List[int] = []
        self.errors: List[str] = []
        self.final_message: Optional[str] = None

        # --- retrieval ---
        self.rewritten_queries: List[str] = []  # [original, HyDE, variation1, ...]
        self.raw_docs: List[Dict] = []
        self.ranked_docs: List[Dict] = []

        # --- ingestion tracking ---
        self.ingestion_done: bool = False   # True once ingestion has run (caps retries at 1)
        self.is_fallback: bool = False      # True when answer is a low-confidence fallback

        # --- answer ---
        self.final_answer: str = ""
        self.confidence: float = 0.0        # context confidence score from AnswerAgent
        self.confidence_cached: bool = False # True when Phase 2c already computed confidence
        self.chat_history: List[Dict] = []  # [{"query": ..., "answer": ...}]

        # --- structured output ---
        self.sources: List[Dict] = []       # [{"title": str, "url": str}]
        self.latency_ms: Dict[str, int] = {
            "retrieve_ms": 0,
            "rerank_ms": 0,
            "llm_ms": 0,
        }
        self.status: str = "success"        # "success" | "low_confidence" | "fallback" | "error"

        # --- decision trace ---
        # Populated progressively during the pipeline; exposed in the API response.
        self.decision_trace: Dict[str, str] = {
            "retrieval_quality":    "",   # filled after rerank
            "action":               "",   # filled at ingestion/fallback decision points
            "confidence_reasoning": "",   # filled after compute_composite
        }

        # Internal signal — NOT exposed in the API response.
        # Strength of the last ingestion decision: "strong" | "moderate" | "weak".
        # Reserved for future reasoning loops (e.g. adaptive retry budgets).
        self._decision_strength: str = ""

        # Self-critique scoring — NOT exposed in the API response.
        self._critique_score:  float = 0.0   # score ∈ [0,1] from critique_answer()
        self._critique_reason: str   = ""    # one-sentence explanation
        self._critique_type:   str   = ""    # "incomplete"|"incorrect"|"not_grounded"|"good"
        self._retried:         bool  = False  # True after one retry attempt
