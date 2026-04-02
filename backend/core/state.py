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

        # --- answer ---
        self.final_answer: str = ""
        self.chat_history: List[Dict] = []  # [{"query": ..., "answer": ...}]
