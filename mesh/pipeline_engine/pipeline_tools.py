"""
Standalone research tools for the grant review pipeline.

Extracted from mesh source code — zero mesh dependency.

Tools provided:
  - exa_search, exa_fetch_full        (Exa web search, needs EXA_API_KEY)
  - arxiv_search, arxiv_get, arxiv_fulltext
  - pubmed_search, pubmed_get, pubmed_fulltext
  - literature_search                  (unified arXiv + PubMed)
  - extract_url                        (PDF/HTML → text)
  - file_read                          (local file with line numbers)
"""

import difflib
import json
import os
import random
import sys
import re
import subprocess
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import logging
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# arXiv Client (from mesh/literature/sources/arxiv.py)
# ---------------------------------------------------------------------------

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    updated: str
    categories: list[str]
    pdf_url: str
    abs_url: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comment: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "updated": self.updated,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "abs_url": self.abs_url,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "comment": self.comment,
        }


class ArxivClient:
    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, delay_seconds: float = 3.0):
        self.delay = delay_seconds
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GrantReviewPipeline/1.0 (research assistant)"
        })

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed + random.uniform(0, 2.0))
        self.last_request_time = time.time()

    def _parse_entry(self, entry: ET.Element) -> ArxivPaper:
        def get_text(tag: str, ns: str = ATOM_NS) -> str:
            el = entry.find(f"{ns}{tag}")
            return el.text.strip() if el is not None and el.text else ""

        entry_id = get_text("id")
        arxiv_id_match = re.search(r"arxiv\.org/abs/([^v]+)", entry_id)
        arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else entry_id

        title = " ".join(get_text("title").split())

        authors = []
        for author in entry.findall(f"{ATOM_NS}author"):
            name = author.find(f"{ATOM_NS}name")
            if name is not None and name.text:
                authors.append(name.text.strip())

        abstract = " ".join(get_text("summary").split())

        categories = []
        for cat in entry.findall(f"{ARXIV_NS}primary_category"):
            term = cat.get("term")
            if term:
                categories.append(term)
        for cat in entry.findall(f"{ATOM_NS}category"):
            term = cat.get("term")
            if term and term not in categories:
                categories.append(term)

        pdf_url = ""
        abs_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
            elif link.get("type") == "text/html":
                abs_url = link.get("href", "")

        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if not abs_url and arxiv_id:
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"

        doi = get_text("doi", ARXIV_NS)
        journal_ref = get_text("journal_ref", ARXIV_NS)
        comment = get_text("comment", ARXIV_NS)

        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            published=get_text("published"),
            updated=get_text("updated"),
            categories=categories,
            pdf_url=pdf_url,
            abs_url=abs_url,
            doi=doi if doi else None,
            journal_ref=journal_ref if journal_ref else None,
            comment=comment if comment else None,
        )

    def search(
        self, query: str, max_results: int = 10,
        sort_by: str = "relevance", sort_order: str = "descending",
        search_field: str = "all",
    ) -> list[ArxivPaper]:
        self._wait_for_rate_limit()
        search_query = f"{search_field}:{query}" if search_field != "all" else query
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        try:
            r = self.session.get(self.BASE_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [arxiv] API error on query '{search_query}': {e}", file=sys.stderr)
            return []
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError:
            return []
        papers = []
        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                papers.append(self._parse_entry(entry))
            except Exception:
                continue
        return papers

    def get_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        self._wait_for_rate_limit()
        params = {"id_list": arxiv_id, "max_results": 1}
        try:
            r = self.session.get(self.BASE_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [arxiv] API error fetching paper {arxiv_id}: {e}", file=sys.stderr)
            return None
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError:
            return None
        entries = root.findall(f"{ATOM_NS}entry")
        if not entries:
            return None
        try:
            return self._parse_entry(entries[0])
        except Exception:
            return None

    def extract_arxiv_id(self, text: str) -> Optional[str]:
        for pattern in [
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",
            r"arxiv:(\d{4}\.\d{4,5})",
            r"\b(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$",
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def download_pdf_text(
        self, paper: ArxivPaper, full_text: bool = True,
        max_pages: Optional[int] = None,
    ) -> Optional[str]:
        if not paper.pdf_url:
            return None
        try:
            r = self.session.get(paper.pdf_url, timeout=60)
            r.raise_for_status()
        except requests.RequestException:
            return None
        if not r.content.startswith(b"%PDF-"):
            return None
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(r.content)
                tmp.flush()
                tmp_path = tmp.name
            cmd = ["pdftotext"]
            if not full_text:
                cmd.extend(["-f", "1", "-l", "1"])
            elif max_pages:
                cmd.extend(["-f", "1", "-l", str(max_pages)])
            cmd.extend([tmp_path, "-"])
            text = subprocess.check_output(cmd, text=True, timeout=60).strip()
            return text if text else None
        except Exception:
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# PubMed Client (from mesh/literature/sources/pubmed.py)
# ---------------------------------------------------------------------------

@dataclass
class PubMedPaper:
    pmid: str
    title: str
    authors: list[str]
    abstract: str
    journal: str
    pub_date: str
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def pmc_url(self) -> Optional[str]:
        if self.pmcid:
            return f"https://pmc.ncbi.nlm.nih.gov/articles/{self.pmcid}/"
        return None

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid, "title": self.title,
            "authors": self.authors, "abstract": self.abstract,
            "journal": self.journal, "pub_date": self.pub_date,
            "doi": self.doi, "pmcid": self.pmcid,
            "keywords": self.keywords, "mesh_terms": self.mesh_terms,
            "pubmed_url": self.pubmed_url, "pmc_url": self.pmc_url,
        }


class PubMedClient:
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    BIOC_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None,
                 rate_limit: float = 0.34):
        self.api_key = api_key
        self.email = email
        self.rate_limit = rate_limit if not api_key else 0.1
        self._last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GrantReviewPipeline/1.0 (research assistant)"
        })

    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _add_auth_params(self, params: dict) -> dict:
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    def search(self, query: str, max_results: int = 10, sort: str = "relevance",
               min_date: Optional[str] = None, max_date: Optional[str] = None,
               ) -> list[PubMedPaper]:
        self._rate_limit_wait()
        search_params = self._add_auth_params({
            "db": "pubmed", "term": query,
            "retmax": max_results, "retmode": "json", "sort": sort,
        })
        if min_date:
            search_params["mindate"] = min_date
        if max_date:
            search_params["maxdate"] = max_date
        if min_date or max_date:
            search_params["datetype"] = "pdat"
        resp = self.session.get(f"{self.EUTILS_BASE}/esearch.fcgi",
                                params=search_params, timeout=30)
        resp.raise_for_status()
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []
        return self._fetch_papers(id_list)

    def _fetch_papers(self, pmids: list[str]) -> list[PubMedPaper]:
        self._rate_limit_wait()
        fetch_params = self._add_auth_params({
            "db": "pubmed", "id": ",".join(pmids),
            "rettype": "xml", "retmode": "xml",
        })
        resp = self.session.get(f"{self.EUTILS_BASE}/efetch.fcgi",
                                params=fetch_params, timeout=60)
        resp.raise_for_status()
        return self._parse_pubmed_xml(resp.text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[PubMedPaper]:
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers
        for article in root.findall(".//PubmedArticle"):
            try:
                p = self._parse_article(article)
                if p:
                    papers.append(p)
            except Exception:
                continue
        return papers

    def _parse_article(self, article: ET.Element) -> Optional[PubMedPaper]:
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None
        pmid_elem = medline.find(".//PMID")
        if pmid_elem is None or pmid_elem.text is None:
            return None
        pmid = pmid_elem.text
        art = medline.find(".//Article")
        if art is None:
            return None

        title_elem = art.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        abstract_parts = []
        abstract_elem = art.find(".//Abstract")
        if abstract_elem is not None:
            for text in abstract_elem.findall(".//AbstractText"):
                label = text.get("Label", "")
                content = text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {content}")
                else:
                    abstract_parts.append(content)
        abstract = " ".join(abstract_parts)

        authors = []
        author_list = art.find(".//AuthorList")
        if author_list is not None:
            for author in author_list.findall(".//Author"):
                lastname = author.find("LastName")
                forename = author.find("ForeName")
                if lastname is not None and lastname.text:
                    name = lastname.text
                    if forename is not None and forename.text:
                        name = f"{forename.text} {name}"
                    authors.append(name)

        journal_elem = art.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None and journal_elem.text else ""

        pub_date_parts = []
        pub_date_elem = art.find(".//Journal/JournalIssue/PubDate")
        if pub_date_elem is not None:
            for tag in ("Year", "Month", "Day"):
                el = pub_date_elem.find(tag)
                if el is not None and el.text:
                    pub_date_parts.append(el.text)
        pub_date = " ".join(pub_date_parts)

        doi = None
        for eloc in art.findall(".//ELocationID"):
            if eloc.get("EIdType") == "doi":
                doi = eloc.text
                break

        pmcid = None
        pubmed_data = article.find(".//PubmedData")
        if pubmed_data is not None:
            for art_id in pubmed_data.findall(".//ArticleId"):
                if art_id.get("IdType") == "pmc":
                    pmcid = art_id.text
                    break

        keywords = []
        keyword_list = medline.find(".//KeywordList")
        if keyword_list is not None:
            for kw in keyword_list.findall(".//Keyword"):
                if kw.text:
                    keywords.append(kw.text)

        mesh_terms = []
        mesh_list = medline.find(".//MeshHeadingList")
        if mesh_list is not None:
            for mesh in mesh_list.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)

        return PubMedPaper(
            pmid=pmid, title=title, authors=authors, abstract=abstract,
            journal=journal, pub_date=pub_date, doi=doi, pmcid=pmcid,
            keywords=keywords, mesh_terms=mesh_terms,
        )

    def get_paper(self, pmid: str) -> Optional[PubMedPaper]:
        papers = self._fetch_papers([pmid])
        return papers[0] if papers else None

    def get_fulltext(self, pmid: Optional[str] = None,
                     pmcid: Optional[str] = None) -> Optional[str]:
        if not pmid and not pmcid:
            return None
        identifier = pmcid if pmcid else pmid
        self._rate_limit_wait()
        url = f"{self.BIOC_BASE}/BioC_json/{identifier}/unicode"
        try:
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._extract_text_from_bioc_json(resp.text)
        except requests.RequestException:
            return None

    def _extract_text_from_bioc_json(self, json_text: str) -> str:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return json_text
        text_parts = []
        collections = data if isinstance(data, list) else [data]
        for collection in collections:
            for doc in collection.get("documents", []):
                for passage in doc.get("passages", []):
                    infons = passage.get("infons", {})
                    section_type = infons.get("section_type", "")
                    text = passage.get("text", "")
                    if text:
                        if section_type:
                            text_parts.append(f"\n[{section_type}]\n{text}")
                        else:
                            text_parts.append(text)
        return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Exa Search Client — REST API (no SDK dependency)
# ---------------------------------------------------------------------------

_EXA_BASE = "https://api.exa.ai"

class ExaSearchClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ""
        if not self.api_key:
            print("[exa] WARNING: EXA_API_KEY not set — exa_search will be unavailable")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def search(self, query: str, num_results: int = 8) -> str:
        if not self.is_available():
            return "Exa API not available (EXA_API_KEY not set)."
        try:
            resp = requests.post(
                f"{_EXA_BASE}/search",
                headers=self._headers(),
                json={
                    "query": query,
                    "numResults": num_results,
                    "contents": {"text": {"maxCharacters": 600}},
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return f"Exa search error: {e}"
        results = data.get("results", [])
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, start=1):
            snippet = (r.get("text") or "").strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(f"{i}. {r.get('title', '(no title)')}\n   {r.get('url', '')}\n   {snippet}\n")
        return "\n".join(lines)

    def search_structured(self, query: str, num_results: int = 8) -> list[dict]:
        """Return raw structured results (list of dicts) instead of formatted text."""
        if not self.is_available():
            return []
        try:
            resp = requests.post(
                f"{_EXA_BASE}/search",
                headers=self._headers(),
                json={
                    "query": query,
                    "numResults": num_results,
                    "contents": {"text": {"maxCharacters": 600}},
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException:
            return []

    def fetch_full_content_by_url(self, url: str) -> str:
        if not self.is_available():
            return "Exa API not available (EXA_API_KEY not set)."
        try:
            resp = requests.post(
                f"{_EXA_BASE}/contents",
                headers=self._headers(),
                json={
                    "ids": [url],
                    "contents": {"text": True},
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return f"Exa full-content error: {e}"
        results = data.get("results", [])
        if not results:
            return "No full content available."
        page = results[0]
        title = page.get("title") or "(no title)"
        body = (page.get("text") or "").strip()
        return f"{title}\n{url}\n\n{body}"


# ---------------------------------------------------------------------------
# URL / PDF / HTML text extraction (from mesh/literature/fulltext_extractor.py)
# ---------------------------------------------------------------------------

def _extract_text_from_url(url: str, max_pages: Optional[int] = None) -> Optional[str]:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
    })

    # Try PDF first
    try:
        r = session.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
    except Exception:
        return None

    content_type = r.headers.get("Content-Type", "").lower()
    is_pdf = "pdf" in content_type or r.content.startswith(b"%PDF-")

    if is_pdf:
        return _extract_from_pdf_bytes(r.content, max_pages=max_pages)

    # Try HTML
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for el in soup.find_all(["script", "style", "nav", "footer", "header"]):
            el.decompose()
        main = (
            soup.find("main") or soup.find("article") or
            soup.find("div", class_=re.compile(r"content|article|paper|body", re.I)) or
            soup.body or soup
        )
        text = main.get_text(separator="\n", strip=True)
        if text:
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()
    except Exception:
        pass

    return None


def _extract_from_pdf_bytes(pdf_bytes: bytes,
                            max_pages: Optional[int] = None) -> Optional[str]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = tmp.name
        cmd = ["pdftotext", "-layout"]
        if max_pages:
            cmd.extend(["-f", "1", "-l", str(max_pages)])
        cmd.extend([tmp_path, "-"])
        text = subprocess.check_output(cmd, text=True, timeout=60).strip()
        if not text:
            return None
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'-\n(\w)', r'\1', text)
        text = text.replace('\f', '\n\n')
        return text.strip()
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Singleton clients (created once on import)
# ---------------------------------------------------------------------------

_arxiv = ArxivClient(delay_seconds=3.0)
_pubmed = PubMedClient()
_exa = ExaSearchClient(api_key=os.environ.get("EXA_API_KEY"))


# ---------------------------------------------------------------------------
# Semantic Scholar — delegates to mesh layer
# ---------------------------------------------------------------------------

_MESH_HOME = os.environ.get("MESH_HOME", str(Path(__file__).resolve().parent.parent))
if _MESH_HOME not in sys.path:
    sys.path.insert(0, _MESH_HOME)

try:
    from mesh.literature.semantic import SemanticScholar as _S2Client
    try:
        _s2 = _S2Client(api_key=os.environ.get("S2_API_KEY"))
    except TypeError:
        # Older mesh clients do not accept api_key; they read auth from env/config.
        _s2 = _S2Client()
except ImportError:
    _s2 = None

# S2 disabled — OpenAlex handles all graph/search/paper endpoints.
_s2 = None

try:
    from mesh.literature.openalex import OpenAlexClient as _OAClient
    _oa = _OAClient()
except ImportError:
    _oa = None

_s2_consecutive_429s = 0
_S2_429_ABORT_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Tool handler functions — each returns a JSON-serialisable value
# ---------------------------------------------------------------------------

def handle_exa_search(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})
    num_results = min(int(args.get("num_results", 8)), 12)
    return _exa.search(query, num_results)


def handle_exa_fetch_full(args: dict) -> str:
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "Missing required parameter: url"})
    return _exa.fetch_full_content_by_url(url)


def handle_arxiv_search(args: dict):
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing required parameter: query"}
    max_results = int(args.get("max_results", 10))
    search_field = args.get("search_field", "all")
    try:
        papers = _arxiv.search(query=query, max_results=max_results,
                               search_field=search_field)
    except Exception as e:
        return {"error": f"arxiv_search failed: {e}"}
    return [
        {
            "arxiv_id": p.arxiv_id, "title": p.title,
            "authors": p.authors[:5],
            "abstract": p.abstract,
            "published": p.published[:10] if p.published else None,
            "categories": p.categories,
            "pdf_url": p.pdf_url, "abs_url": p.abs_url,
        }
        for p in papers
    ]


def handle_arxiv_get(args: dict):
    arxiv_id = args.get("arxiv_id", "").strip()
    if not arxiv_id:
        url = args.get("url", "").strip()
        if url:
            arxiv_id = _arxiv.extract_arxiv_id(url)
    if not arxiv_id:
        return {"error": "Missing required parameter: arxiv_id or url"}
    try:
        paper = _arxiv.get_paper(arxiv_id)
    except Exception as e:
        return {"error": f"arxiv_get failed: {e}"}
    if not paper:
        return {"error": f"Paper not found: {arxiv_id}"}
    return {
        "arxiv_id": paper.arxiv_id, "title": paper.title,
        "authors": paper.authors, "abstract": paper.abstract,
        "published": paper.published[:10] if paper.published else None,
        "categories": paper.categories,
        "pdf_url": paper.pdf_url, "abs_url": paper.abs_url,
        "doi": paper.doi, "journal_ref": paper.journal_ref,
    }


def handle_arxiv_fulltext(args: dict):
    arxiv_id = args.get("arxiv_id", "").strip()
    if not arxiv_id:
        url = args.get("url", "").strip()
        if url:
            arxiv_id = _arxiv.extract_arxiv_id(url)
    if not arxiv_id:
        return {"error": "Missing required parameter: arxiv_id or url"}
    max_pages = args.get("max_pages")
    if max_pages:
        max_pages = int(max_pages)
    try:
        paper = _arxiv.get_paper(arxiv_id)
        if not paper:
            return {"error": f"Paper not found: {arxiv_id}"}
        text = _arxiv.download_pdf_text(paper, full_text=True, max_pages=max_pages)
    except Exception as e:
        return {"error": f"arxiv_fulltext failed: {e}"}
    if not text:
        return {"error": f"Could not extract text from paper: {arxiv_id}"}
    return {
        "arxiv_id": paper.arxiv_id, "title": paper.title,
        "text": text, "char_count": len(text),
    }


def handle_literature_fulltext(args: dict):
    arxiv_id = (args.get("arxiv_id") or "").strip()
    doi = (args.get("doi") or "").strip()
    pmid = (args.get("pmid") or "").strip()
    max_pages = args.get("max_pages")

    if arxiv_id:
        return handle_arxiv_fulltext({"arxiv_id": arxiv_id, "max_pages": max_pages})

    if pmid:
        return handle_pubmed_fulltext({"pmid": pmid})

    if doi:
        try:
            url = f"https://doi.org/{doi}"
            content = _exa.fetch_full_content_by_url(url)
            if content and not isinstance(content, dict):
                return {"doi": doi, "text": content, "char_count": len(content)}
            if isinstance(content, dict) and content.get("text"):
                return {"doi": doi, "text": content["text"], "char_count": len(content["text"])}
        except Exception:
            pass

    return {"error": "Could not fetch fulltext. Provide arxiv_id, doi, or pmid."}


def handle_pubmed_search(args: dict):
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing required parameter: query"}
    max_results = int(args.get("max_results", 10))
    sort = args.get("sort", "relevance")
    min_date = args.get("min_date")
    max_date = args.get("max_date")
    try:
        papers = _pubmed.search(query=query, max_results=max_results,
                                sort=sort, min_date=min_date, max_date=max_date)
    except Exception as e:
        return {"error": f"pubmed_search failed: {e}"}
    return [
        {
            "pmid": p.pmid, "title": p.title,
            "authors": p.authors[:5],
            "abstract": p.abstract[:1000] if p.abstract else None,
            "journal": p.journal, "pub_date": p.pub_date,
            "doi": p.doi, "pmcid": p.pmcid,
            "pubmed_url": p.pubmed_url, "pmc_url": p.pmc_url,
        }
        for p in papers
    ]


def handle_pubmed_get(args: dict):
    pmid = args.get("pmid", "").strip()
    if not pmid:
        return {"error": "Missing required parameter: pmid"}
    try:
        paper = _pubmed.get_paper(pmid)
    except Exception as e:
        return {"error": f"pubmed_get failed: {e}"}
    if not paper:
        return {"error": f"Paper not found: {pmid}"}
    return {
        "pmid": paper.pmid, "title": paper.title,
        "authors": paper.authors, "abstract": paper.abstract,
        "journal": paper.journal, "pub_date": paper.pub_date,
        "doi": paper.doi, "pmcid": paper.pmcid,
        "keywords": paper.keywords, "mesh_terms": paper.mesh_terms,
        "pubmed_url": paper.pubmed_url, "pmc_url": paper.pmc_url,
    }


def handle_pubmed_fulltext(args: dict):
    pmid = args.get("pmid", "").strip() if args.get("pmid") else None
    pmcid = args.get("pmcid", "").strip() if args.get("pmcid") else None
    if not pmid and not pmcid:
        return {"error": "Missing required parameter: pmid or pmcid"}
    try:
        text = _pubmed.get_fulltext(pmid=pmid, pmcid=pmcid)
    except Exception as e:
        return {"error": f"pubmed_fulltext failed: {e}"}
    if not text:
        return {"error": "Full text not available (may not be in PMC Open Access Subset)."}
    result: dict[str, Any] = {"text": text, "char_count": len(text)}
    if pmid:
        paper = _pubmed.get_paper(pmid)
        if paper:
            result["pmid"] = paper.pmid
            result["title"] = paper.title
            result["pmcid"] = paper.pmcid
    return result


def _s2_to_unified(results: list[dict]) -> list[dict]:
    """Normalize S2 results to the unified paper format used by literature_search."""
    out = []
    for r in results:
        out.append({
            "title": r.get("title", ""),
            "authors": [a.get("name", "") for a in (r.get("authors") or [])[:5]],
            "abstract": (r.get("abstract") or "")[:800] or None,
            "year": r.get("year"),
            "source": "s2",
            "doi": (r.get("externalIds") or {}).get("DOI"),
            "arxiv_id": (r.get("externalIds") or {}).get("ArXiv"),
            "pmid": (r.get("externalIds") or {}).get("PubMed"),
            "url": r.get("url"),
        })
    return out


def _oa_to_unified(results: list[dict]) -> list[dict]:
    """Normalize OpenAlex results to the unified paper format."""
    out = []
    for r in results:
        authors = r.get("author") or r.get("authors") or []
        out.append({
            "title": r.get("title", ""),
            "authors": authors[:5],
            "abstract": (r.get("abstract") or "")[:800] or None,
            "year": r.get("year"),
            "source": "openalex",
            "doi": r.get("doi"),
            "arxiv_id": r.get("arxiv_id"),
            "pmid": None,
            "url": r.get("publisher_url") or r.get("url"),
        })
    return out


def _exa_to_unified(results: list[dict]) -> list[dict]:
    """Normalize exa search results to the unified paper format."""
    out = []
    for r in results:
        out.append({
            "title": r.get("title", "(no title)"),
            "authors": [],
            "abstract": (r.get("text") or "").strip()[:800] or None,
            "year": None,
            "source": "exa",
            "doi": None,
            "arxiv_id": None,
            "pmid": None,
            "url": r.get("url"),
        })
    return out


def handle_s2_search(args: dict):
    """Search Semantic Scholar with automatic cascade to OpenAlex → exa on failure."""
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing required parameter: query"}
    limit = int(args.get("max_results", args.get("limit", 10)))
    year = args.get("year")
    fos = args.get("fields_of_study")
    if isinstance(fos, str):
        fos = [fos]

    s2_error = None
    results = []

    # Try S2 first
    if _s2 is not None:
        try:
            results = _s2.search(query, limit=limit, year=year, fields_of_study=fos)
        except Exception as e:
            s2_error = str(e)

    if results:
        unified = _s2_to_unified(results)
        return {"query": query, "count": len(unified), "results": unified, "source": "s2"}

    # S2 returned empty (likely 429 exhaustion) or errored — cascade to OpenAlex
    if _oa is not None:
        try:
            oa_kwargs: dict[str, Any] = {"max_results": limit}
            if year:
                oa_kwargs["date_from"] = f"{year}-01-01"
                oa_kwargs["date_to"] = f"{year}-12-31"
            oa_results = _oa.search(query, **oa_kwargs)
            if oa_results:
                unified = _oa_to_unified(oa_results)
                return {
                    "query": query, "count": len(unified), "results": unified,
                    "source": "openalex", "fallback_reason": s2_error or "s2_empty",
                }
        except Exception as e:
            logger.debug(f"OpenAlex fallback failed: {e}")

    # OpenAlex also failed — try exa
    if _exa and _exa.is_available():
        try:
            exa_results = _exa.search_structured(query, num_results=limit)
            if exa_results:
                unified = _exa_to_unified(exa_results)
                return {
                    "query": query, "count": len(unified), "results": unified,
                    "source": "exa", "fallback_reason": s2_error or "s2_and_oa_empty",
                }
        except Exception as e:
            logger.debug(f"Exa fallback failed: {e}")

    if s2_error:
        return {"error": f"All search sources failed. S2: {s2_error}"}
    return {"query": query, "count": 0, "results": [], "source": "all_failed"}


def handle_s2_get(args: dict):
    """Get detailed paper info from Semantic Scholar, with OpenAlex fallback."""
    paper_id = args.get("paper_id", "").strip()
    if not paper_id:
        return {"error": "Missing required parameter: paper_id"}

    # Try S2 first (unless persistently rate-limited)
    if _s2 is not None and _s2_consecutive_429s < _S2_429_ABORT_THRESHOLD:
        try:
            result = _s2.get_paper(paper_id)
            if result:
                return result
        except Exception as e:
            if "429" not in str(e) or _oa is None:
                return {"error": f"S2 get failed: {e}"}

    # Fallback to OpenAlex
    if _oa is not None:
        doi = None
        arxiv_id = None
        if paper_id.startswith("DOI:"):
            doi = paper_id[4:]
        elif paper_id.startswith("arXiv:"):
            arxiv_id = paper_id[6:]
        if doi:
            result = _oa.get_by_doi(doi)
            if result:
                result["source"] = "openalex"
                return result
        elif arxiv_id:
            results = _oa.search(arxiv_id, max_results=1)
            if results:
                results[0]["source"] = "openalex"
                return results[0]

    return {"error": "S2 and OpenAlex both failed for paper lookup"}


def handle_s2_citations(args: dict):
    """Get papers that cite a given paper."""
    if _s2 is None:
        return {"error": "S2 not available — set MESH_HOME or install mesh package"}
    paper_id = args.get("paper_id", "").strip()
    if not paper_id:
        return {"error": "Missing required parameter: paper_id"}
    limit = int(args.get("max_results", args.get("limit", 50)))
    try:
        results = _s2.get_citations(paper_id, limit=limit)
        return {"paper_id": paper_id, "count": len(results), "citations": results}
    except Exception as e:
        return {"error": f"S2 citations failed: {e}"}


def handle_s2_references(args: dict):
    """Get papers referenced by a given paper."""
    if _s2 is None:
        return {"error": "S2 not available — set MESH_HOME or install mesh package"}
    paper_id = args.get("paper_id", "").strip()
    if not paper_id:
        return {"error": "Missing required parameter: paper_id"}
    limit = int(args.get("max_results", args.get("limit", 50)))
    try:
        results = _s2.get_references(paper_id, limit=limit)
        return {"paper_id": paper_id, "count": len(results), "references": results}
    except Exception as e:
        return {"error": f"S2 references failed: {e}"}


def handle_openalex_search(args: dict):
    """Search OpenAlex for academic papers (~250M works, 10 req/s)."""
    if _oa is None:
        return {"error": "OpenAlex not available — set MESH_HOME or install mesh package"}
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing required parameter: query"}
    max_results = int(args.get("max_results", 20))
    date_from = args.get("date_from")
    date_to = args.get("date_to")
    try:
        results = _oa.search(query, max_results=max_results, date_from=date_from, date_to=date_to)
        return {"query": query, "count": len(results), "results": results}
    except Exception as e:
        return {"error": f"OpenAlex search failed: {e}"}


# ---------------------------------------------------------------------------
# Graph-first discovery handlers — replace bulk keyword search with targeted
# reference/citation traversal for dramatically fewer S2 calls.
# ---------------------------------------------------------------------------

_CROSSREF_HEADERS = {
    "User-Agent": "MeshPipeline/1.0 (mailto:user@example.com)"
}


def _crossref_extract_metadata(item: dict) -> dict:
    """Extract unified metadata from a CrossRef work item."""
    authors = []
    for a in (item.get("author") or [])[:10]:
        name = f"{a.get('given', '')} {a.get('family', '')}".strip()
        if name:
            authors.append(name)

    year = None
    for date_field in ["published-print", "published-online", "created"]:
        parts = (item.get(date_field) or {}).get("date-parts", [[]])
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break

    return {
        "title": (item.get("title") or [""])[0],
        "authors": authors,
        "year": year,
        "doi": item.get("DOI"),
        "abstract": (item.get("abstract") or "")[:800] or None,
        "url": item.get("URL"),
        "source": "crossref",
        "resolved": True,
        "container_title": (item.get("container-title") or [""])[0],
        "type": item.get("type"),
        "score": item.get("score"),
    }


def handle_crossref_resolve(args: dict):
    """Resolve a paper reference via CrossRef API. Returns structured metadata."""
    title = args.get("title", "").strip()
    doi = args.get("doi", "").strip()
    if not title and not doi:
        return {"error": "Missing required parameter: title or doi"}

    try:
        if doi:
            url = f"https://api.crossref.org/works/{requests.utils.quote(doi, safe='')}"
            resp = requests.get(url, timeout=15, headers=_CROSSREF_HEADERS)
            if resp.status_code == 404:
                return {"doi": doi, "resolved": False, "reason": "doi_not_in_crossref"}
            if resp.status_code != 200:
                return {"error": f"CrossRef returned HTTP {resp.status_code}"}
            return _crossref_extract_metadata(resp.json().get("message", {}))
        else:
            resp = requests.get(
                "https://api.crossref.org/works", timeout=15,
                params={"query.bibliographic": title, "rows": 3},
                headers=_CROSSREF_HEADERS,
            )
            if resp.status_code != 200:
                return {"error": f"CrossRef returned HTTP {resp.status_code}"}

            items = resp.json().get("message", {}).get("items", [])
            if not items:
                return {"title": title, "resolved": False, "reason": "no_match"}

            best = items[0]
            result = _crossref_extract_metadata(best)

            # Validate: check that the resolved title is reasonably close to the query
            resolved_title = result.get("title", "").lower()
            query_lower = title.lower()
            # Use SequenceMatcher for fuzzy title matching
            ratio = difflib.SequenceMatcher(None, query_lower[:100], resolved_title[:100]).ratio()
            if ratio < 0.4 and (best.get("score") or 0) < 30:
                return {"title": title, "resolved": False, "reason": "low_confidence",
                        "best_candidate": result.get("title"), "match_ratio": round(ratio, 2)}

            result["match_ratio"] = round(ratio, 2)
            return result
    except Exception as e:
        return {"title": title or doi, "resolved": False, "error": str(e)}


def handle_references_extract(args: dict):
    """Extract and resolve references from a paper's bibliography section.

    Input: list of reference strings (raw bibliography entries).
    Output: resolved metadata for each reference via CrossRef.
    """
    references = args.get("references", [])
    if isinstance(references, str):
        references = [r.strip() for r in references.split("\n") if r.strip()]
    if not references:
        return {"error": "Missing required parameter: references (list of bibliography strings)"}

    max_refs = int(args.get("max_refs", 30))
    references = references[:max_refs]

    resolved = []
    failed = []
    for ref_text in references:
        # Try to extract a DOI from the reference string
        doi_match = re.search(r'10\.\d{4,}/[^\s,;}\]]+', ref_text)
        if doi_match:
            result = handle_crossref_resolve({"doi": doi_match.group().rstrip('.')})
        else:
            # Strip reference number prefix
            cleaned = re.sub(r'^\s*\[?\d+\]?\s*', '', ref_text)
            # Use query.bibliographic with the full reference text — CrossRef
            # handles mixed author+title+venue queries well via this field
            query = cleaned[:200]
            result = handle_crossref_resolve({"title": query})

        if result.get("resolved"):
            result["original_text"] = ref_text[:200]
            resolved.append(result)
        else:
            failed.append({"text": ref_text[:200], "reason": result.get("reason", result.get("error", "unknown"))})

        time.sleep(0.1)  # polite rate limiting for CrossRef

    return {
        "total_references": len(references),
        "resolved": len(resolved),
        "failed": len(failed),
        "papers": resolved,
        "unresolved": failed[:10],
    }


def _citation_expand_via_openalex(doi, arxiv_id, max_citations, max_references):
    """Fallback citation/reference expansion using OpenAlex when S2 is rate-limited."""
    result: dict[str, Any] = {"s2_calls": 0, "source": "openalex"}

    try:
        citing = _oa.get_cited_by(doi=doi, arxiv_id=arxiv_id, max_results=max_citations)
        result["citing_papers"] = _oa_to_unified(citing)
        result["citing_count"] = len(citing)
    except Exception as e:
        result["citing_papers"] = []
        result["citing_count"] = 0
        result["citations_error"] = str(e)

    try:
        refs = _oa.get_references(doi=doi, arxiv_id=arxiv_id, max_results=max_references)
        result["referenced_papers"] = _oa_to_unified(refs)
        result["referenced_count"] = len(refs)
    except Exception as e:
        result["referenced_papers"] = []
        result["referenced_count"] = 0
        result["references_error"] = str(e)

    result["recommended_papers"] = []
    result["recommended_count"] = 0

    total = result.get("citing_count", 0) + result.get("referenced_count", 0)
    result["total_discovered"] = total
    return result


def handle_citation_expand(args: dict):
    """Targeted graph traversal for a paper — citations, references, and recommendations.

    Tries S2 first; on rate-limit (429), falls back to OpenAlex for citation/reference data.
    Tracks consecutive S2 failures and skips S2 entirely after threshold.
    """
    global _s2_consecutive_429s

    paper_id = args.get("paper_id", "").strip()
    doi = args.get("doi", "").strip()
    if not paper_id and not doi:
        return {"error": "Missing required parameter: paper_id or doi"}

    if not paper_id and doi:
        paper_id = f"DOI:{doi}"

    # Extract doi/arxiv_id from paper_id for OpenAlex fallback
    _doi = doi
    _arxiv_id = None
    if paper_id.startswith("DOI:"):
        _doi = paper_id[4:]
    elif paper_id.startswith("arXiv:"):
        _arxiv_id = paper_id[6:]

    max_citations = int(args.get("max_citations", 20))
    max_references = int(args.get("max_references", 20))

    # If S2 is persistently rate-limited, go straight to OpenAlex
    if _s2_consecutive_429s >= _S2_429_ABORT_THRESHOLD and _oa is not None:
        logger.info("S2 rate-limited (%d consecutive 429s) — using OpenAlex for %s",
                     _s2_consecutive_429s, paper_id)
        result = _citation_expand_via_openalex(_doi, _arxiv_id, max_citations, max_references)
        result["paper_id"] = paper_id
        result["fallback_reason"] = f"s2_rate_limited_{_s2_consecutive_429s}_consecutive"
        return result

    if _s2 is None:
        if _oa is not None:
            result = _citation_expand_via_openalex(_doi, _arxiv_id, max_citations, max_references)
            result["paper_id"] = paper_id
            result["fallback_reason"] = "s2_unavailable"
            return result
        return {"error": "Neither S2 nor OpenAlex client available"}

    result: dict[str, Any] = {"paper_id": paper_id, "s2_calls": 0, "source": "s2"}
    s2_got_429 = False
    paper = None

    # Call 1: Get the paper's own metadata
    try:
        paper = _s2.get_paper(paper_id)
        result["s2_calls"] += 1
        if paper:
            result["paper"] = {
                "title": paper.get("title"),
                "year": paper.get("year"),
                "citation_count": paper.get("citationCount"),
                "reference_count": paper.get("referenceCount"),
            }
            _s2_consecutive_429s = 0
    except Exception as e:
        err_str = str(e)
        result["paper_error"] = err_str
        if "429" in err_str:
            s2_got_429 = True
            _s2_consecutive_429s += 1

    # If S2 is 429ing and OpenAlex is available, fall back immediately
    if s2_got_429 and _oa is not None:
        logger.info("S2 429 on get_paper for %s — falling back to OpenAlex", paper_id)
        oa_result = _citation_expand_via_openalex(_doi, _arxiv_id, max_citations, max_references)
        oa_result["paper_id"] = paper_id
        oa_result["fallback_reason"] = "s2_429_on_get_paper"
        oa_result["s2_calls"] = result["s2_calls"]
        return oa_result

    # Call 2: Get citing papers (who builds on this work)
    try:
        citations = _s2.get_citations(paper_id, limit=max_citations)
        result["s2_calls"] += 1
        result["citing_papers"] = _s2_to_unified(citations)
        result["citing_count"] = len(citations)
        if citations:
            _s2_consecutive_429s = 0
    except Exception as e:
        err_str = str(e)
        result["citations_error"] = err_str
        result["citing_papers"] = []
        result["citing_count"] = 0
        if "429" in err_str:
            _s2_consecutive_429s += 1

    # Call 3: Get referenced papers (what this work builds on)
    try:
        references = _s2.get_references(paper_id, limit=max_references)
        result["s2_calls"] += 1
        result["referenced_papers"] = _s2_to_unified(references)
        result["referenced_count"] = len(references)
        if references:
            _s2_consecutive_429s = 0
    except Exception as e:
        err_str = str(e)
        result["references_error"] = err_str
        result["referenced_papers"] = []
        result["referenced_count"] = 0
        if "429" in err_str:
            _s2_consecutive_429s += 1

    # Call 4 (optional): Recommended papers via the recommendations API
    try:
        s2_id = paper_id
        if paper and paper.get("paperId"):
            s2_id = paper["paperId"]
        resp = requests.get(
            f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{s2_id}",
            params={"fields": "paperId,title,year,authors,abstract,url,citationCount,externalIds", "limit": 10},
            headers={"x-api-key": _s2.session.headers.get("x-api-key", "")},
            timeout=15,
        )
        result["s2_calls"] += 1
        if resp.status_code == 200:
            recs = resp.json().get("recommendedPapers", [])
            result["recommended_papers"] = _s2_to_unified(recs)
            result["recommended_count"] = len(recs)
        else:
            result["recommended_papers"] = []
            result["recommended_count"] = 0
            result["recommended_note"] = f"HTTP {resp.status_code}"
    except Exception as e:
        result["recommended_papers"] = []
        result["recommended_count"] = 0
        result["recommended_error"] = str(e)

    total = result.get("citing_count", 0) + result.get("referenced_count", 0) + result.get("recommended_count", 0)
    result["total_discovered"] = total
    return result


def handle_unpaywall_lookup(args: dict):
    """Look up open-access PDF links for a DOI via Unpaywall API (free, high rate limits)."""
    doi = args.get("doi", "").strip()
    if not doi:
        return {"error": "Missing required parameter: doi"}

    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{requests.utils.quote(doi, safe='')}",
            params={"email": os.environ.get("OPENALEX_EMAIL", "user@example.com")},
            timeout=15,
        )
        if resp.status_code == 404:
            return {"doi": doi, "found": False, "reason": "DOI not in Unpaywall database"}
        if resp.status_code != 200:
            return {"doi": doi, "found": False, "error": f"HTTP {resp.status_code}"}

        data = resp.json()
        best_oa = data.get("best_oa_location") or {}
        oa_locations = data.get("oa_locations") or []

        pdf_urls = []
        for loc in oa_locations:
            pdf = loc.get("url_for_pdf") or loc.get("url")
            if pdf:
                pdf_urls.append({
                    "url": pdf,
                    "host_type": loc.get("host_type"),
                    "version": loc.get("version"),
                })

        return {
            "doi": doi,
            "found": True,
            "is_oa": data.get("is_oa", False),
            "title": data.get("title"),
            "journal": data.get("journal_name"),
            "year": data.get("year"),
            "best_pdf_url": (best_oa.get("url_for_pdf") or best_oa.get("url")) if best_oa else None,
            "all_pdf_urls": pdf_urls[:5],
            "oa_status": data.get("oa_status"),
        }
    except Exception as e:
        return {"doi": doi, "found": False, "error": str(e)}


def handle_literature_search(args: dict):
    """Unified search across arXiv + PubMed with deduplication."""
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing required parameter: query"}
    max_results = int(args.get("max_results", 10))

    sources_arg = args.get("sources")
    search_arxiv = True
    search_pubmed = True
    if sources_arg:
        names = sources_arg if isinstance(sources_arg, list) else [sources_arg]
        search_arxiv = "arxiv" in names
        search_pubmed = "pubmed" in names

    all_papers: list[dict] = []

    if search_arxiv:
        try:
            arxiv_papers = _arxiv.search(query, max_results=max_results)
            for p in arxiv_papers:
                year = None
                if p.published:
                    m = re.match(r"(\d{4})", p.published)
                    if m:
                        year = int(m.group(1))
                all_papers.append({
                    "title": p.title, "authors": p.authors[:5],
                    "abstract": p.abstract[:800] if p.abstract else None,
                    "year": year, "source": "arxiv",
                    "doi": p.doi, "arxiv_id": p.arxiv_id,
                    "pmid": None, "pmcid": None,
                    "pdf_url": p.pdf_url, "publisher_url": p.abs_url,
                })
        except Exception:
            pass

    if search_pubmed:
        try:
            pm_papers = _pubmed.search(query, max_results=max_results)
            for p in pm_papers:
                year = None
                if p.pub_date:
                    m = re.match(r"(\d{4})", p.pub_date)
                    if m:
                        year = int(m.group(1))
                all_papers.append({
                    "title": p.title, "authors": p.authors[:5],
                    "abstract": p.abstract[:800] if p.abstract else None,
                    "year": year, "source": "pubmed",
                    "doi": p.doi, "arxiv_id": None,
                    "pmid": p.pmid, "pmcid": p.pmcid,
                    "pdf_url": p.pmc_url, "publisher_url": p.pubmed_url,
                })
        except Exception:
            pass

    # Deduplicate by DOI or title similarity
    seen_dois: set[str] = set()
    seen_titles: list[str] = []
    deduped: list[dict] = []
    for paper in all_papers:
        if paper.get("doi"):
            d = paper["doi"].lower()
            if d in seen_dois:
                continue
            seen_dois.add(d)
        norm = re.sub(r"[^\w\s]", " ", (paper.get("title") or "").lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if any(difflib.SequenceMatcher(None, norm, t).ratio() > 0.85
               for t in seen_titles):
            continue
        seen_titles.append(norm)
        deduped.append(paper)

    return {"query": query, "count": len(deduped), "results": deduped[:max_results * 2]}


def handle_extract_url(args: dict):
    url = args.get("url", "").strip()
    if not url:
        return {"error": "Missing required parameter: url"}
    max_pages = args.get("max_pages")
    if max_pages:
        max_pages = int(max_pages)
    try:
        text = _extract_text_from_url(url, max_pages=max_pages)
    except Exception as e:
        return {"error": f"Extraction failed: {e}"}
    if not text:
        return {"error": "Could not extract text from URL"}
    return {
        "url": url, "text": text,
        "char_count": len(text), "estimated_tokens": len(text) // 4,
    }


def _resolve_home(path: str) -> str:
    """Expand ~ to the real user home from /etc/passwd, ignoring $HOME.

    Mirrors mesh.paths.resolve_path — vendored here because this module is
    loaded both as a top-level module (engine dir on sys.path) and as
    mesh.pipeline_engine.pipeline_tools. $HOME is a synthetic CC acct home
    when the process was launched from a CC session.
    """
    import pwd
    path = str(path)
    real = pwd.getpwuid(os.getuid()).pw_dir
    if path == "~":
        return real
    if path.startswith("~/"):
        return os.path.join(real, path[2:])
    if path.startswith("~"):
        return os.path.expanduser(path)
    return path


def handle_file_read(args: dict) -> str:
    path = args.get("path", "").strip()
    if not path:
        return "Error: Missing required parameter: path"
    path = _resolve_home(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"
    total = len(lines)
    start_line = int(args.get("start_line", 1))
    num_lines = int(args.get("num_lines", 200))
    end_line = args.get("end_line")
    start_idx = max(0, start_line - 1)
    end_from_num = start_idx + num_lines
    if end_line is not None:
        end_idx = min(total, end_from_num, int(end_line))
    else:
        end_idx = min(total, end_from_num)
    numbered = []
    for i, line in enumerate(lines[start_idx:end_idx], start=start_idx + 1):
        numbered.append(f"{i:4d}|{line.rstrip(chr(10))}")
    return "\n".join(numbered) + f"\n\n({total} lines total)"


def handle_grep(args: dict) -> str:
    """Search files for a pattern using ripgrep (rg) or grep."""
    import subprocess as _sp
    pattern = args.get("pattern", "").strip()
    if not pattern:
        return "Error: Missing required parameter: pattern"
    path = args.get("path", ".").strip()
    path = _resolve_home(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        return f"Error: Path not found: {path}"

    include = args.get("include", "")
    max_count = int(args.get("max_results", 50))
    case_insensitive = bool(args.get("case_insensitive", False))

    import shutil
    for tool_name in ["rg", "grep"]:
        tool_path = shutil.which(tool_name)
        if not tool_path:
            for fallback in [f"/usr/bin/{tool_name}", f"/usr/local/bin/{tool_name}"]:
                if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
                    tool_path = fallback
                    break
        if not tool_path:
            continue
        if tool_name == "rg":
            cmd = [tool_path, "-n", "--max-count", str(max_count), "--max-columns", "200"]
            if case_insensitive:
                cmd.append("-i")
            if include:
                cmd.extend(["--glob", include])
            cmd.extend([pattern, path])
        else:
            cmd = [tool_path, "-rn", f"--max-count={max_count}"]
            if case_insensitive:
                cmd.append("-i")
            if include:
                cmd.extend(["--include", include])
            cmd.extend([pattern, path])
        try:
            result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
            out = result.stdout.strip()
            if not out and result.returncode != 0:
                continue
            lines = out.split("\n")
            if len(lines) > max_count:
                lines = lines[:max_count]
                out = "\n".join(lines) + f"\n[... truncated at {max_count} results]"
            return out or "No matches found."
        except FileNotFoundError:
            continue
        except _sp.TimeoutExpired:
            return f"Error: grep timed out after 30s"
    return "Error: neither rg nor grep found on PATH"


# ---------------------------------------------------------------------------
# OpenAI function-calling tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": "Search the web using Exa. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "num_results": {"type": "integer", "description": "Number of results (default 8, max 12)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exa_fetch_full",
            "description": "Fetch the full text content of a URL via Exa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch full content for."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": "Search arXiv for papers. Returns titles, abstracts, arXiv IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "max_results": {"type": "integer", "description": "Max results (default 10)."},
                    "search_field": {
                        "type": "string",
                        "description": "Field to search: 'all', 'ti' (title), 'au' (author), 'abs' (abstract).",
                        "enum": ["all", "ti", "au", "abs", "cat"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_get",
            "description": "Fetch a specific arXiv paper by ID. Returns full metadata and abstract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID (e.g. '1706.03762')."},
                    "url": {"type": "string", "description": "arXiv URL (alternative to arxiv_id)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_fulltext",
            "description": "Download an arXiv paper PDF and extract the full text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID."},
                    "url": {"type": "string", "description": "arXiv URL (alternative to arxiv_id)."},
                    "max_pages": {"type": "integer", "description": "Limit extraction to first N pages."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Search PubMed for biomedical literature. Supports PubMed query syntax.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (supports PubMed syntax)."},
                    "max_results": {"type": "integer", "description": "Max results (default 10)."},
                    "sort": {
                        "type": "string",
                        "description": "Sort order.",
                        "enum": ["relevance", "pub_date", "first_author"],
                    },
                    "min_date": {"type": "string", "description": "Minimum date (YYYY or YYYY/MM/DD)."},
                    "max_date": {"type": "string", "description": "Maximum date (YYYY or YYYY/MM/DD)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_get",
            "description": "Fetch a specific PubMed paper by PMID. Returns full metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {"type": "string", "description": "PubMed ID (numeric string)."},
                },
                "required": ["pmid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_fulltext",
            "description": "Get full text from PMC (PubMed Central). Only works for Open Access papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {"type": "string", "description": "PubMed ID."},
                    "pmcid": {"type": "string", "description": "PMC ID (e.g. 'PMC1234567'), preferred if available."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "literature_search",
            "description": (
                "Unified literature search across arXiv and PubMed with deduplication. "
                "Automatically searches both sources and merges results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "max_results": {"type": "integer", "description": "Max results per source (default 10)."},
                    "sources": {
                        "description": "Optional: list of sources to search ('arxiv', 'pubmed'). Defaults to both.",
                        "oneOf": [
                            {"type": "string", "enum": ["arxiv", "pubmed"]},
                            {"type": "array", "items": {"type": "string", "enum": ["arxiv", "pubmed"]}},
                        ],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "s2_search",
            "description": (
                "Search Semantic Scholar for academic papers. Returns metadata, "
                "abstract, citation count, and external IDs (arXiv, DOI, PubMed). "
                "Faster and more reliable than arXiv search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "limit": {"type": "integer", "description": "Max results (default 10, max 100)."},
                    "year": {"type": "string", "description": "Year filter, e.g. '2023', '2020-2024', '2023-'."},
                    "fields_of_study": {
                        "description": "Filter by field(s): Computer Science, Medicine, Physics, etc.",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "s2_get",
            "description": (
                "Get detailed paper metadata from Semantic Scholar, including "
                "full reference list and a sample of citing papers. Accepts S2 paper ID, "
                "DOI (DOI:10.xxx), arXiv ID (arXiv:1706.03762), or PMID (PMID:123)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "S2 paper ID or external ID with prefix (DOI:, arXiv:, PMID:)."},
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "s2_citations",
            "description": "Get papers that cite a given paper. Useful for tracing impact and follow-up work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "S2 paper ID or external ID with prefix."},
                    "limit": {"type": "integer", "description": "Max citations to return (default 50, max 1000)."},
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "s2_references",
            "description": "Get papers referenced by a given paper. Useful for understanding a paper's foundation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "S2 paper ID or external ID with prefix."},
                    "limit": {"type": "integer", "description": "Max references to return (default 50, max 1000)."},
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_url",
            "description": "Extract full text from a PDF or HTML URL. Works with arXiv, publisher sites, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract text from."},
                    "max_pages": {"type": "integer", "description": "Limit PDF extraction to first N pages."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openalex_search",
            "description": (
                "Search OpenAlex for academic papers across all disciplines (~250M works). "
                "Supports date-range filtering. Higher rate limits than S2 (10 req/s vs 1 req/s). "
                "Use for broad keyword sweeps, date-filtered searches, and when S2 is rate-limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "max_results": {"type": "integer", "description": "Max results (default 20, max 200)."},
                    "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)."},
                    "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crossref_resolve",
            "description": (
                "Resolve a paper reference via CrossRef API (free, high rate limits). "
                "Given a title or DOI, returns structured metadata: title, authors, year, DOI, abstract."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Paper title or bibliographic query string."},
                    "doi": {"type": "string", "description": "DOI to resolve directly (preferred over title if available)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "references_extract",
            "description": (
                "Extract and resolve references from a paper's bibliography. "
                "Input a list of raw bibliography strings; each is resolved via CrossRef. "
                "Returns structured metadata for each resolved reference. "
                "Use this instead of keyword search to discover the paper's citation neighborhood."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "references": {
                        "description": "List of raw bibliography entry strings from the paper.",
                        "oneOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "string", "description": "Newline-separated bibliography entries."},
                        ],
                    },
                    "max_refs": {"type": "integer", "description": "Maximum references to resolve (default 30)."},
                },
                "required": ["references"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "citation_expand",
            "description": (
                "Graph-first discovery: given a paper ID or DOI, returns citing papers, "
                "referenced papers, and recommended similar papers via S2 graph endpoints. "
                "Uses only 3-4 targeted S2 API calls (not keyword search), so it bypasses "
                "the rate limiting that plagues s2_search. Much more reliable for finding "
                "competing and related work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "S2 paper ID or prefixed external ID (DOI:10.xxx, arXiv:1706.03762)."},
                    "doi": {"type": "string", "description": "DOI (used if paper_id not provided)."},
                    "max_citations": {"type": "integer", "description": "Max citing papers to return (default 20)."},
                    "max_references": {"type": "integer", "description": "Max referenced papers to return (default 20)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unpaywall_lookup",
            "description": (
                "Look up open-access PDF links for a paper via Unpaywall (free, high rate limits). "
                "Given a DOI, returns the best available PDF URL and all open-access locations. "
                "Use this to find fulltext PDFs without going through S2 or publisher paywalls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {"type": "string", "description": "Paper DOI (e.g. '10.1145/1234567.1234568')."},
                },
                "required": ["doi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a local file with line numbers. Useful for reading proposals, papers, and data files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "start_line": {"type": "integer", "description": "Starting line number (default 1)."},
                    "num_lines": {"type": "integer", "description": "Number of lines to read (default 200)."},
                    "end_line": {"type": "integer", "description": "End line number (inclusive)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a regex pattern. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {"type": "string", "description": "Directory or file to search in (default: current directory)."},
                    "include": {"type": "string", "description": "File glob pattern, e.g. '*.py' or '*.{py,js}'."},
                    "max_results": {"type": "integer", "description": "Maximum number of matching lines (default 50)."},
                    "case_insensitive": {"type": "boolean", "description": "Case-insensitive search (default false)."},
                },
                "required": ["pattern"],
            },
        },
    },
]

# Dispatch table: tool name -> handler function
TOOL_HANDLERS: dict[str, callable] = {
    "exa_search": handle_exa_search,
    "exa_fetch_full": handle_exa_fetch_full,
    "arxiv_search": handle_arxiv_search,
    "arxiv_get": handle_arxiv_get,
    "arxiv_fulltext": handle_arxiv_fulltext,
    "pubmed_search": handle_pubmed_search,
    "pubmed_get": handle_pubmed_get,
    "pubmed_fulltext": handle_pubmed_fulltext,
    "literature_search": handle_literature_search,
    "s2_search": handle_s2_search,
    "crossref_resolve": handle_crossref_resolve,
    "references_extract": handle_references_extract,
    "citation_expand": handle_citation_expand,
    "unpaywall_lookup": handle_unpaywall_lookup,
    "s2_get": handle_s2_get,
    "s2_citations": handle_s2_citations,
    "s2_references": handle_s2_references,
    "openalex_search": handle_openalex_search,
    "literature_fulltext": handle_literature_fulltext,
    "extract_url": handle_extract_url,
    "file_read": handle_file_read,
    "grep": handle_grep,
}


def dispatch_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return the result as a string."""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        result = handler(arguments)
    except Exception as e:
        result = {"error": f"{name} raised: {e}"}
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2)
