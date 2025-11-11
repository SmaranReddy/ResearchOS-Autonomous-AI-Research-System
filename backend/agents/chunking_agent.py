# agents/chunking_agent.py
class ChunkingAgent:
    """Splits text into overlapping chunks for embedding."""

    def __init__(self, chunk_size=1500, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str):
        chunks, start = [], 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks
