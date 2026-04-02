# ==========================================
# backend/utils/tavily_utils.py
# Tavily Search + PDF Downloader
# ==========================================

import os
import requests
import json
from io import BytesIO
from PyPDF2 import PdfReader


class TavilyClient:
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.headers = {"Content-Type": "application/json"}
        self.search_url = "https://api.tavily.com/search"
        self.download_dir = "downloads"
        os.makedirs(self.download_dir, exist_ok=True)
        print("✅ Tavily client initialized successfully")

    def _extract_text(self, pdf_bytes):
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()

    def search_and_download(self, query, num_papers=5):
        """Search Tavily for PDFs and download them."""
        payload = {
            "query": query,
            "max_results": num_papers,
            "include_domains": ["arxiv.org", "ieee.org", "acm.org"]
        }

        response = requests.post(self.search_url, headers=self.headers, json=payload)
        data = response.json().get("results", [])
        print(f"📄 Tavily returned {len(data)} paper results with PDFs.")

        papers = []
        for item in data:
            title = item.get("title", "Untitled Paper")
            pdf_url = item.get("url")
            if not pdf_url or not pdf_url.endswith(".pdf"):
                print(f"❌ Skipping non-PDF: {title}")
                continue

            try:
                pdf_data = requests.get(pdf_url, timeout=20)
                pdf_data.raise_for_status()
                text = self._extract_text(pdf_data.content)

                pdf_name = title[:80].replace("/", "_").strip() + ".pdf"
                json_name = pdf_name.replace(".pdf", ".json")

                pdf_path = os.path.join(self.download_dir, pdf_name)
                json_path = os.path.join(self.download_dir, json_name)

                with open(pdf_path, "wb") as f:
                    f.write(pdf_data.content)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({"title": title, "url": pdf_url, "content": text}, f, indent=2)

                print(f"📄 Saved PDF: {pdf_path}")
                print(f"💾 Saved JSON: {json_path}")

                papers.append({"title": title, "path": pdf_path, "content": text})
            except Exception as e:
                print(f"❌ Could not download PDF from {pdf_url}: {e}")

        return papers
