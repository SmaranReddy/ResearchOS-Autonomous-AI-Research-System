# agents/sanitize_agent.py
import re

class SanitizeAgent:
    """Cleans and sanitizes filenames or text identifiers."""

    @staticmethod
    def sanitize_filename(title: str) -> str:
        """Removes invalid characters and extra spaces."""
        title = re.sub(r'[\\/*?:"<>|]', "", title)
        title = title.replace("\n", " ").replace("\r", " ").strip()
        return " ".join(title.split())
