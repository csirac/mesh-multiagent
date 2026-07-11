import logging
import os
import requests
import time
from urllib.parse import quote

logger = logging.getLogger(__name__)


class OpenAlexClient:
    BASE_URL = "https://api.openalex.org/works"
    MIN_REQUEST_INTERVAL = 0.1
    _RETRY_BACKOFFS = [5, 15, 30]

    def __init__(self, email=None, timeout=15):
        if email is None:
            email = os.environ.get("OPENALEX_EMAIL", "user@example.com")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = f"mailto:{email}"
        self._last_request_time = 0.0

    def _throttled_get(self, url):
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()
        resp = self.session.get(url, timeout=self.timeout)
        for attempt, backoff in enumerate(self._RETRY_BACKOFFS, 1):
            if resp.status_code != 429:
                break
            logger.warning("OpenAlex rate-limited (429) on %s — retry %d/%d in %ds",
                           url[:120], attempt, len(self._RETRY_BACKOFFS), backoff)
            time.sleep(backoff)
            self._last_request_time = time.monotonic()
            resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code == 429:
            logger.error("OpenAlex rate limit persisted after %d retries — returning empty",
                         len(self._RETRY_BACKOFFS))
            return resp
        resp.raise_for_status()
        return resp

    @staticmethod
    def _reconstruct_abstract(inverted_index):
        if not inverted_index or not isinstance(inverted_index, dict):
            return None
        words = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        words.sort(key=lambda x: x[0])
        return " ".join(w for _, w in words) if words else None

    @staticmethod
    def _extract_authors(authorships):
        if not authorships or not isinstance(authorships, list):
            return []
        authors = []
        for a in authorships:
            author = a.get("author", {})
            name = author.get("display_name")
            if name:
                authors.append(name)
        return authors

    @staticmethod
    def _parse_ids(work):
        ids = work.get("ids") or {}
        arxiv_id = None
        doi = None

        openalex_id = ids.get("openalex", "")
        raw_doi = ids.get("doi") or work.get("doi") or ""
        if raw_doi.startswith("https://doi.org/"):
            doi = raw_doi[len("https://doi.org/"):]
        elif raw_doi:
            doi = raw_doi

        raw_arxiv = ids.get("openalex", "")
        for key in ["arxiv", "openalex"]:
            val = ids.get(key, "")
            if "arxiv.org" in val:
                parts = val.rstrip("/").split("/")
                arxiv_id = parts[-1] if parts else None
                break

        return arxiv_id, doi

    def _normalize_work(self, work):
        arxiv_id, doi = self._parse_ids(work)

        primary_loc = work.get("primary_location") or {}
        pdf_url = None
        publisher_url = None
        if isinstance(primary_loc, dict):
            pdf_url = primary_loc.get("pdf_url")
            publisher_url = primary_loc.get("landing_page_url")

        oa = work.get("open_access") or {}
        if not pdf_url and isinstance(oa, dict):
            pdf_url = oa.get("oa_url")

        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))

        return {
            "author": self._extract_authors(work.get("authorships")),
            "title": work.get("title") or work.get("display_name"),
            "year": work.get("publication_year"),
            "abstract": abstract,
            "arxiv_id": arxiv_id or "",
            "doi": doi or "",
            "s2_id": "",
            "pdf_url": pdf_url,
            "publisher_url": publisher_url,
            "source": "openalex",
        }

    def search(self, query, max_results=20, date_from=None, date_to=None):
        filters = []
        if date_from:
            filters.append(f"from_publication_date:{date_from}")
        if date_to:
            filters.append(f"to_publication_date:{date_to}")

        url = f"{self.BASE_URL}?search={quote(query)}&per_page={min(max_results, 200)}"
        if filters:
            url += f"&filter={','.join(filters)}"

        resp = self._throttled_get(url)
        if resp.status_code == 429:
            return []

        data = resp.json()
        results = []
        for work in data.get("results", []):
            if len(results) >= max_results:
                break
            results.append(self._normalize_work(work))
        return results

    def get_by_doi(self, doi):
        clean = doi.strip()
        if clean.startswith("https://doi.org/"):
            clean = clean[len("https://doi.org/"):]
        url = f"https://api.openalex.org/works/doi:{quote(clean, safe='')}"
        resp = self._throttled_get(url)
        if resp.status_code == 429:
            return {}
        if resp.status_code == 404:
            return {}
        return self._normalize_work(resp.json())

    def _resolve_openalex_id(self, doi=None, arxiv_id=None):
        """Resolve a DOI or arXiv ID to an OpenAlex work ID."""
        if doi:
            clean = doi.strip()
            if clean.startswith("https://doi.org/"):
                clean = clean[len("https://doi.org/"):]
            url = f"https://api.openalex.org/works/doi:{quote(clean, safe='')}"
        elif arxiv_id:
            clean = arxiv_id.strip().replace("arXiv:", "")
            url = f"https://api.openalex.org/works?search={quote(clean)}&per_page=1"
        else:
            return None, {}

        try:
            resp = self._throttled_get(url)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None, {}
            raise
        if resp.status_code in (429, 404):
            return None, {}

        data = resp.json()
        if "results" in data:
            works = data.get("results", [])
            if not works:
                return None, {}
            work = works[0]
        else:
            work = data

        oa_id = work.get("id", "")
        return oa_id, work

    def get_cited_by(self, doi=None, arxiv_id=None, max_results=20):
        """Get papers that cite the given work (by DOI or arXiv ID)."""
        oa_id, _ = self._resolve_openalex_id(doi=doi, arxiv_id=arxiv_id)
        if not oa_id:
            return []

        oa_short = oa_id.replace("https://openalex.org/", "")
        url = (f"{self.BASE_URL}?filter=cites:{oa_short}"
               f"&per_page={min(max_results, 200)}")
        resp = self._throttled_get(url)
        if resp.status_code == 429:
            return []

        data = resp.json()
        results = []
        for work in data.get("results", []):
            if len(results) >= max_results:
                break
            results.append(self._normalize_work(work))
        return results

    def get_references(self, doi=None, arxiv_id=None, max_results=20):
        """Get papers referenced by the given work (by DOI or arXiv ID)."""
        oa_id, work_data = self._resolve_openalex_id(doi=doi, arxiv_id=arxiv_id)
        if not oa_id:
            return []

        ref_ids = work_data.get("referenced_works", [])
        if not ref_ids:
            return []

        ref_ids = ref_ids[:max_results]
        oa_shorts = "|".join(
            rid.replace("https://openalex.org/", "") for rid in ref_ids
        )
        url = f"{self.BASE_URL}?filter=ids.openalex:{oa_shorts}&per_page={min(len(ref_ids), 200)}"
        resp = self._throttled_get(url)
        if resp.status_code == 429:
            return []

        data = resp.json()
        results = []
        for work in data.get("results", []):
            results.append(self._normalize_work(work))
        return results
