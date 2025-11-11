# agents/preprocessing_agent.py
import re

class PreprocessingAgent:
    """Cleans extracted PDF text before tokenization."""
    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\n+", "\n", text)
        text = re.split(r"\bReferences\b", text, maxsplit=1)[0]
        text = re.sub(r"Figure\s*\d+|Table\s*\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
