import requests
import os, re, json
from io import BytesIO
from time import sleep
from PyPDF2 import PdfReader

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}

class Downloader:
    def __init__(self, save_dir="downloads"):
        os.makedirs(save_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)
        self.save_dir = save_dir

    def _sanitize_filename(self, title):
        safe = re.sub(r'[\\/*?:"<>|]', "", title or "untitled")
        return " ".join(safe.split())

    def download_and_extract(self, item: dict) -> dict:
        url = item.get("link", "")
        title = item.get("title", "untitled")
        safe_title = self._sanitize_filename(title)
        pdf_path = os.path.join(self.save_dir, safe_title + ".pdf")
        json_path = os.path.join(self.save_dir, safe_title + ".json")

        if not url:
            item["full_text"] = ""
            return item

        try:
            pdf_bytes = self._attempt_pdf_download(url)
            if not pdf_bytes:
                print(f"❌ Could not download PDF: {url}")
                item["full_text"] = ""
                return item

            # Save PDF
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"📄 Saved PDF: {pdf_path}")

            # Extract text
            text = self._extract_text(pdf_bytes)
            item["full_text"] = text

            # Save metadata JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "title": title,
                    "url": url,
                    "chars": len(text),
                    "preview": text[:1000]
                }, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved JSON: {json_path}")

        except Exception as e:
            print(f"⚠️ Download or extraction failed: {e}")
            item["full_text"] = ""
        return item

    def _attempt_pdf_download(self, url):
        try_urls = [url]

        # ✅ Springer fix
        if "springer.com/article/" in url:
            doi = url.split("/article/")[-1]
            try_urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")

        # ✅ IEEE fix
        if "ieeexplore.ieee.org/document/" in url:
            doc_id = re.findall(r"/document/(\d+)", url)
            if doc_id:
                try_urls.append(f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={doc_id[0]}")

        # ✅ ACM fix
        if "dl.acm.org/doi/" in url:
            try_urls.append(url.replace("/doi/", "/doi/pdf/"))

        # ✅ arXiv fix
        if "arxiv.org/abs/" in url:
            try_urls.append(url.replace("abs", "pdf") + ".pdf")

        # Attempt download with retries
        for attempt in range(3):
            for u in try_urls:
                try:
                    r = self.session.get(u, timeout=25, allow_redirects=True)
                    if "application/pdf" in r.headers.get("Content-Type", "") or r.content.startswith(b"%PDF"):
                        return r.content
                except Exception as e:
                    print(f"⚠️ Retry failed ({e})")
            sleep(1)
        return None

    def _extract_text(self, pdf_bytes):
        text = ""
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"⚠️ PDF parse error: {e}")
        return text.strip()
