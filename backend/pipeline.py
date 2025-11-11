# backend/pipeline.py

import re
import time
import numpy as np
from agents.arxiv_agent import ArxivAgent
from agents.downloader_agent import DownloaderAgent
from agents.embedder_agent import EmbedderAgent
from agents.index_agent import IndexAgent
from agents.llm_agent import SummarizerAgent


# ---------------------------------------------------------------------
# 🧹 Helper: Sanitize filenames for Windows
# ---------------------------------------------------------------------
def sanitize_filename(title: str) -> str:
    """Removes invalid characters and extra spaces from filenames."""
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    title = title.replace("\n", " ").replace("\r", " ").strip()
    return " ".join(title.split())  # normalize multiple spaces


# ---------------------------------------------------------------------
# 🧰 Preprocessing Agent
# ---------------------------------------------------------------------
class PreprocessingAgent:
    """Cleans raw text extracted from PDF before embedding."""
    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        import re
        # Remove excessive newlines and spaces
        text = re.sub(r"\n+", "\n", text)
        # Remove references section if present
        text = re.split(r"\bReferences\b", text, maxsplit=1)[0]
        # Remove figure/table mentions
        text = re.sub(r"Figure\s*\d+|Table\s*\d+", "", text)
        # Normalize spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ---------------------------------------------------------------------
# 🧮 Tokenizer Agent
# ---------------------------------------------------------------------
class TokenizerAgent:
    """Tokenizes text to count or truncate tokens."""
    def __init__(self):
        try:
            import tiktoken
            self.tiktoken = tiktoken
            self.enc = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception:
            self.tiktoken = None
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())

    def truncate(self, text: str, max_tokens: int = 8000) -> str:
        if self.enc:
            tokens = self.enc.encode(text)
            truncated = tokens[:max_tokens]
            return self.enc.decode(truncated)
        # fallback: truncate by words
        return " ".join(text.split()[:max_tokens])


# ---------------------------------------------------------------------
# ✂️ Chunking Agent
# ---------------------------------------------------------------------
class ChunkingAgent:
    """Splits text into overlapping chunks for embedding."""
    def __init__(self, chunk_size: int = 1500, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks


# ---------------------------------------------------------------------
# 🚀 Main RAG Pipeline
# ---------------------------------------------------------------------
def main(query: str, num_papers: int = 1):
    """Run the full research pipeline for a given query."""
    print("=" * 100)
    print(f"🧠 Starting Research Pipeline for query: '{query}'")
    print("=" * 100)

    # Initialize all agents
    print("\n⚙️ Initializing agents...")
    arxiv = ArxivAgent()
    downloader = DownloaderAgent()
    embedder = EmbedderAgent()
    indexer = IndexAgent()
    summarizer = SummarizerAgent()
    preprocessor = PreprocessingAgent()
    tokenizer = TokenizerAgent()
    chunker = ChunkingAgent()
    print("✅ All agents initialized successfully.\n")

    # Step 1: Fetch papers
    print(f"🔍 Searching ArXiv for: '{query}' ...")
    papers = arxiv.fetch_papers(query, max_results=num_papers)
    print(f"📚 Found {len(papers)} papers from ArXiv.\n")

    for i, paper in enumerate(papers, start=1):
        print("=" * 100)
        print(f"🧾 Processing Paper {i}/{len(papers)}: {paper['title']}")
        print("=" * 100)
        start_time = time.time()

        try:
            # Step 1: Summarize abstract
            print("✏️ Summarizing abstract using LLM...")
            summary = summarizer.summarize(paper["abstract"])
            paper["summary_structured"] = summary
            print("✅ Abstract summarized successfully.")
            print(f"📝 Summary preview:\n{summary[:400]}...\n")

            # Step 2: Sanitize title for filenames
            safe_title = sanitize_filename(paper["title"])
            paper["safe_title"] = safe_title
            print(f"🧩 Sanitized title for saving: '{safe_title}'")

            # Step 3: Download and extract text
            print("📥 Downloading and extracting full text from PDF...")
            paper = downloader.download_and_extract(paper)
            if not paper.get("full_text"):
                print("⚠️ No full text extracted. Skipping embedding.")
                continue

            full_text = paper["full_text"]
            print(f"📄 Full text extracted ({len(full_text)} characters).")

            # Step 4: Preprocess text
            clean_text = preprocessor.preprocess(full_text)
            print(f"🧹 Cleaned text length: {len(clean_text)} chars.")

            # Step 5: Tokenize
            token_count = tokenizer.count_tokens(clean_text)
            print(f"🧮 Token count: {token_count}")

            # Optional truncate if too large
            if token_count > 8000:
                clean_text = tokenizer.truncate(clean_text)
                print("⚠️ Text truncated to 8000 tokens for safety.")

            # Step 6: Chunk text
            chunks = chunker.chunk_text(clean_text)
            print(f"✂️ Split into {len(chunks)} chunks for embedding.\n")

            # Step 7: Embed and index chunks
            for j, chunk in enumerate(chunks, start=1):
                embedding = embedder.embed_text(chunk)
                vec = np.array(embedding)
                print(f"📊 Chunk {j}: Embedding dim {vec.shape[0]}")
                metadata = {
                    **paper,
                    "chunk_id": j,
                    "chunk_text": chunk[:400] + "...",  # small preview
                }
                resp = indexer.upsert_paper(metadata, embedding)
                print(f"✅ Uploaded chunk {j} to Pinecone. Resp: {str(resp)[:100]}")

            # Step 8: Save summary locally
            summary_path = f"{safe_title}.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"💾 Saved structured summary as: {summary_path}")

            duration = round(time.time() - start_time, 2)
            print(f"⏱️ Completed processing '{safe_title}' in {duration} seconds.\n")

        except Exception as e:
            print(f"❌ Error while processing '{paper['title']}': {e}")
            continue

    print("=" * 100)
    print("🚀 RAG Pipeline completed successfully for all papers.")
    print("=" * 100)


# ---------------------------------------------------------------------
# 🧑‍💻 Entry Point (Interactive mode)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    user_query = input("Enter your research query: ").strip()
    if not user_query:
        print("⚠️ No query entered. Exiting.")
    else:
        try:
            num = int(input("Enter number of papers to retrieve: ").strip() or "1")
        except ValueError:
            num = 1
        main(user_query, num)
