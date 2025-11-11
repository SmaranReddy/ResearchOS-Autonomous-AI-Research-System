import tiktoken

class TokenizerAgent:
    """Tokenizes text to estimate length and truncate if necessary."""
    def __init__(self, model_name="text-embedding-3-small"):
        self.enc = tiktoken.encoding_for_model(model_name)

    def tokenize(self, text: str):
        tokens = self.enc.encode(text)
        return tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def truncate(self, text: str, max_tokens: int = 8000) -> str:
        tokens = self.tokenize(text)
        truncated = tokens[:max_tokens]
        return self.enc.decode(truncated)
