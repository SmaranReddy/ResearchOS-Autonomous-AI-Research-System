import requests
import fitz  # PyMuPDF
import os
import re
import json

class DownloaderAgent:
    def __init__(self, save_dir="downloads"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def _sanitize_filename(self, title: str) -> str:
        """Remove invalid characters and normalize spaces."""
        safe = re.sub(r'[\\/*?:"<>|]', "", title)
        safe = safe.replace("\n", " ").replace("\r", " ").strip()
        return " ".join(safe.split())

    def download_and_extract(self, paper: dict) -> dict:
        """Download the paper PDF, extract text, and save a JSON summary."""
        pdf_url = paper["link"].replace("abs", "pdf")
        safe_title = self._sanitize_filename(paper["title"])
        pdf_path = os.path.join(self.save_dir, f"{safe_title}.pdf")
        json_path = os.path.join(self.save_dir, f"{safe_title}.json")

        try:
            # 1. Download PDF
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"📥 Downloaded: {pdf_path}")

            # 2. Extract text
            full_text = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    full_text.append(page.get_text())
            paper["full_text"] = "\n".join(full_text)
            print(f"📄 Extracted {len(paper['full_text'])} characters.")

            # 3. Save JSON file
            json_data = {
                "title": paper.get("title"),
                "link": paper.get("link"),
                "full_text": paper.get("full_text"),
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved paper data as JSON: {json_path}")

        except Exception as e:
            print(f"❌ Failed to process {safe_title}: {e}")
            paper["full_text"] = None

        return paper
