# backend/pipeline.py

import os
import json
import time
import traceback

from backend.agents.downloader_agent import DownloaderAgent
from backend.agents.embedder_agent import EmbedderAgent
from backend.agents.index_agent import IndexAgent
from backend.agents.llm_agent import SummarizerAgent
from backend.agents.tavily_agent import fetch_paper_pdf
from backend.agents.sanitize_agent import SanitizeAgent
from backend.agents.preprocessing_agent import PreprocessingAgent
from backend.agents.tokenizer_agent import TokenizerAgent
from backend.agents.chunking_agent import ChunkingAgent



# ---------------------------------------------------------------------
# 🚀 Tavily Research Pipeline
# ---------------------------------------------------------------------
def main(query: str):
    print("=" * 100)
    print(f"🧠 Starting Tavily Research Pipeline for query: '{query}'")
    print("=" * 100)

    # Initialize agents
    print("\n⚙️ Initializing agents...")
    downloader = DownloaderAgent()
    embedder = EmbedderAgent()
    indexer = IndexAgent()
    summarizer = SummarizerAgent()
    sanitizer = SanitizeAgent()
    preprocessor = PreprocessingAgent()
    tokenizer = TokenizerAgent()
    chunker = ChunkingAgent()
    print("✅ All agents initialized successfully.\n")

    # Step 1: Tavily search
    print(f"🌐 Searching Tavily for: '{query}' ...")
    tavily_pdf_link = fetch_paper_pdf(query, max_results=5)

    if not tavily_pdf_link:
        print("❌ No suitable PDF found by Tavily.")
        return

    print(f"📘 Proceeding with Tavily PDF: {tavily_pdf_link}")

    try:
        # Step 2: Download
        print("📥 Downloading and extracting full text...")
        paper = {"title": query, "link": tavily_pdf_link, "abstract": ""}
        paper = downloader.download_and_extract(paper)
        if not paper.get("full_text"):
            print("⚠️ No text extracted.")
            return

        full = paper["full_text"]
        print(f"📄 Extracted {len(full)} characters.")

        # Save raw
        os.makedirs("downloads", exist_ok=True)
        json_path = os.path.join("downloads", f"{sanitizer.sanitize_filename(query)}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(paper, jf, ensure_ascii=False, indent=2)
        print(f"💾 Saved paper JSON: {json_path}")

        # Step 3: Preprocess
        clean = preprocessor.preprocess(full)
        print(f"🧹 Cleaned text length: {len(clean)} chars.")

        # Step 4: Tokenize
        tokens = tokenizer.count_tokens(clean)
        print(f"🧮 Tokens: {tokens}")
        if tokens > 8000:
            clean = tokenizer.truncate(clean)
            print("⚠️ Truncated to 8000 tokens.")

        # Step 5: Chunk
        chunks = chunker.chunk_text(clean)
        print(f"✂️ Split into {len(chunks)} chunks.\n")

        # Step 6: Summarize
        print("✏️ Summarizing document introduction...")
        summary = summarizer.summarize(clean[:2000])
        paper["summary_structured"] = summary
        print(f"✅ Summary created:\n{summary[:300]}...\n")

        # Step 7: Embed & Index
        for i, chunk in enumerate(chunks, 1):
            emb = embedder.embed_text(chunk)
            meta = {
                "title": paper["title"],
                "link": paper["link"],
                "chunk_id": i,
                "summary": paper["summary_structured"][:300],
                "chunk_text": chunk[:400] + "..." if len(chunk) > 400 else chunk,
            }
            resp = indexer.upsert_paper(meta, emb)
            print(f"✅ Upserted chunk {i}. Resp: {str(resp)[:150]}")

        print("\n🚀 Tavily Research Pipeline completed successfully!\n")

    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------
# 🧑‍💻 Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    query = input("Enter your research query: ").strip()
    if not query:
        print("⚠️ No query entered. Exiting.")
    else:
        main(query)
