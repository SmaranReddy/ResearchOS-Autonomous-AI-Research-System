import re

class PreprocessingAgent:
    """Cleans raw text extracted from PDF before embedding."""
    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Remove references section
        text = re.split(r'\bReferences\b', text, maxsplit=1)[0]
        # Remove figure/table mentions
        text = re.sub(r'Figure\s*\d+|Table\s*\d+', '', text)
        # Trim extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
