import requests
import os, re, json, time as _time
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

        _t_dl_start = _time.monotonic()
        try:
            pdf_bytes = self._attempt_pdf_download(url)
            _dl_ms = int((_time.monotonic() - _t_dl_start) * 1000)
            if not pdf_bytes:
                print(f"[download] '{title[:60]}' — FAILED in {_dl_ms}ms  url={url}")
                item["full_text"] = ""
                return item

            print(f"[download] '{title[:60]}' — HTTP download took {_dl_ms}ms")

            # Save PDF
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            # Extract text
            _t_parse = _time.monotonic()
            text = self._extract_text(pdf_bytes)
            _parse_ms = int((_time.monotonic() - _t_parse) * 1000)
            print(f"[download] '{title[:60]}' — PDF parse took {_parse_ms}ms  ({len(text)} chars)")
            item["full_text"] = text

            _total_ms = int((_time.monotonic() - _t_dl_start) * 1000)
            print(f"[download] '{title[:60]}' took {_total_ms}ms total")

            # Save metadata JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "title": title,
                    "url": url,
                    "chars": len(text),
                    "preview": text[:1000]
                }, f, indent=2, ensure_ascii=False)

        except Exception as e:
            _total_ms = int((_time.monotonic() - _t_dl_start) * 1000)
            print(f"[WARN] Download or extraction failed for '{title[:60]}': {e}  ({_total_ms}ms)")
            item["full_text"] = ""
        return item

    def _attempt_pdf_download(self, url):
        try_urls = [url]

        # [OK] Springer fix
        if "springer.com/article/" in url:
            doi = url.split("/article/")[-1]
            try_urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")

        # [OK] IEEE fix
        if "ieeexplore.ieee.org/document/" in url:
            doc_id = re.findall(r"/document/(\d+)", url)
            if doc_id:
                try_urls.append(f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={doc_id[0]}")

        # [OK] ACM fix
        if "dl.acm.org/doi/" in url:
            try_urls.append(url.replace("/doi/", "/doi/pdf/"))

        # [OK] arXiv fix
        if "arxiv.org/abs/" in url:
            try_urls.append(url.replace("abs", "pdf") + ".pdf")

        # Single pass — no retries to keep ingestion fast
        for u in try_urls:
            _t0 = _time.monotonic()
            try:
                r = self.session.get(u, timeout=5, allow_redirects=True)
                _ms = int((_time.monotonic() - _t0) * 1000)
                if "application/pdf" in r.headers.get("Content-Type", "") or r.content.startswith(b"%PDF"):
                    print(f"[download] HTTP GET {u[:80]} → {_ms}ms  ({len(r.content)} bytes)")
                    return r.content
                else:
                    print(f"[download] HTTP GET {u[:80]} → {_ms}ms  (not a PDF, skipping)")
            except Exception as e:
                _ms = int((_time.monotonic() - _t0) * 1000)
                print(f"[WARN] Download failed ({e})  ({_ms}ms)  url={u[:80]}")
        return None

    def _extract_text(self, pdf_bytes):
        text = ""
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"[WARN] PDF parse error: {e}")
        return text.strip()
