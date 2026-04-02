# agents/tokenizer_agent.py
class TokenizerAgent:
    """Token counter and truncation utility."""

    def __init__(self):
        try:
            import tiktoken
            self.enc = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception:
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())

    def truncate(self, text: str, max_tokens: int = 8000) -> str:
        if self.enc:
            tokens = self.enc.encode(text)
            return self.enc.decode(tokens[:max_tokens])
        return " ".join(text.split()[:max_tokens])
