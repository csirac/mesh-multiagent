
import logging
import requests
import json
import time
from urllib.parse import quote

logger = logging.getLogger(__name__)

class SemanticScholar:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
    MIN_REQUEST_INTERVAL = 1.1
    # S2 is disabled for the review pipeline; if this client is called
    # accidentally, fail fast instead of adding 30s of retry backoff.
    _RETRY_BACKOFFS = []

    def __init__(self, timeout=10, api_key=None):
        self.timeout = timeout
        self.session = requests.Session()
        self._last_request_time = 0.0
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def _throttled_get(self, url):
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()
        resp = self.session.get(url, timeout=self.timeout)
        for attempt, backoff in enumerate(self._RETRY_BACKOFFS, 1):
            if resp.status_code != 429:
                break
            logger.warning("S2 rate-limited (429) on %s — retry %d/%d in %ds",
                           url[:120], attempt, len(self._RETRY_BACKOFFS), backoff)
            time.sleep(backoff)
            self._last_request_time = time.monotonic()
            resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code == 429:
            logger.error("S2 rate limit persisted after %d retries (total backoff %ds) — returning empty",
                         len(self._RETRY_BACKOFFS), sum(self._RETRY_BACKOFFS))
            return resp
        resp.raise_for_status()
        return resp

    def search(self, query, limit=20, year=None, fields_of_study=None):
        """
        Search Semantic Scholar using the /bulk endpoint.
        Returns list of dicts with: author, title, year, abstract, pdfurl.
        """
        fields = "title,year,authors,abstract,url,openAccessPdf,externalIds"

        url = f"{self.BASE_URL}?query={quote(query)}&fields={fields}&limit={limit}"
        if year:
            url += f"&year={quote(str(year))}"
        if fields_of_study:
            fos_str = ",".join(fields_of_study) if isinstance(fields_of_study, list) else fields_of_study
            url += f"&fieldsOfStudy={quote(fos_str)}"

        resp = self._throttled_get(url)
        if resp.status_code == 429:
            return []
        r = resp.json()

        # Check for API error response
        if "error" in r:
            raise RuntimeError(f"Semantic Scholar API error: {r.get('message', r['error'])}")

        results = []
        retrieved = 0

        while True:
            if "data" in r:
                for p in r["data"]:
                    if len(results) >= limit:
                        return results

                    pdf_info = p.get("openAccessPdf")
                    pdf_url = pdf_info.get("url") if isinstance(pdf_info, dict) else None

                    ext_ids = p.get("externalIds") or {}
                    entry = {
                        "author": [a.get("name") for a in p.get("authors", [])],
                        "title": p.get("title"),
                        "year": p.get("year"),
                        "abstract": p.get("abstract"),
                        "pdf_url": pdf_url,
                        "publisher_url": p.get("url"),
                        "s2_id": p.get("paperId", ""),
                    }
                    if ext_ids.get("ArXiv"):
                        entry["arxiv_id"] = ext_ids["ArXiv"]
                    if ext_ids.get("DOI"):
                        entry["doi"] = ext_ids["DOI"]
                    results.append(entry)

                retrieved += len(r["data"])

            if "token" not in r or len(results) >= limit:
                break

            next_url = f"{url}&token={r['token']}"
            resp = self._throttled_get(next_url)
            r = resp.json()

        return results

    def get_paper(self, paper_id):
        fields = "paperId,title,year,authors,abstract,url,openAccessPdf,citationCount,referenceCount,externalIds,fieldsOfStudy,publicationDate,journal,venue"
        url = f"{self.PAPER_URL}/{quote(paper_id, safe=':')}?fields={fields}"
        resp = self._throttled_get(url)
        if resp.status_code == 429:
            return {}
        return resp.json()

    def get_citations(self, paper_id, limit=50):
        fields = "paperId,title,year,authors,abstract,url,citationCount"
        url = f"{self.PAPER_URL}/{quote(paper_id, safe=':')}/citations?fields={fields}&limit={min(limit, 1000)}"
        resp = self._throttled_get(url)
        if resp.status_code == 429:
            return []
        data = resp.json().get("data", [])
        return [item.get("citingPaper", item) for item in data]

    def get_references(self, paper_id, limit=50):
        fields = "paperId,title,year,authors,abstract,url,citationCount"
        url = f"{self.PAPER_URL}/{quote(paper_id, safe=':')}/references?fields={fields}&limit={min(limit, 1000)}"
        resp = self._throttled_get(url)
        if resp.status_code == 429:
            return []
        data = resp.json().get("data", [])
        return [item.get("citedPaper", item) for item in data]
