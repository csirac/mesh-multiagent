"""
arXiv API client for literature search.

arXiv provides a free, well-documented API that returns Atom XML.
Most ML/CS papers are available here with full PDFs.

API docs: https://info.arxiv.org/help/api/basics.html
"""

import re
import time
import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from dataclasses import dataclass
from typing import Optional

import requests


# arXiv Atom XML namespace
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


@dataclass
class ArxivPaper:
    """Represents an arXiv paper with metadata and content."""
    arxiv_id: str  # e.g., "1706.03762"
    title: str
    authors: list[str]
    abstract: str
    published: str  # ISO date string
    updated: str
    categories: list[str]  # e.g., ["cs.CL", "cs.LG"]
    pdf_url: str
    abs_url: str  # Abstract page URL
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
    """
    Client for arXiv API.

    Features:
    - Search by query (title, author, abstract, all)
    - Fetch paper by arXiv ID
    - Download and extract full PDF text
    - Rate limiting (3 second delay between requests per API guidelines)
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, delay_seconds: float = 3.0):
        """
        Args:
            delay_seconds: Delay between API requests (arXiv asks for 3s)
        """
        self.delay = delay_seconds
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "LiteratureSearch/1.0 (research assistant)"
        })

    def _wait_for_rate_limit(self):
        """Ensure we wait between requests per arXiv API guidelines."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()

    def _parse_entry(self, entry: ET.Element) -> ArxivPaper:
        """Parse a single Atom entry into an ArxivPaper."""

        def get_text(tag: str, ns: str = ATOM_NS) -> str:
            el = entry.find(f"{ns}{tag}")
            return el.text.strip() if el is not None and el.text else ""

        # Extract arXiv ID from the entry ID URL
        # Format: http://arxiv.org/abs/1706.03762v5
        entry_id = get_text("id")
        arxiv_id_match = re.search(r"arxiv\.org/abs/([^v]+)", entry_id)
        arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else entry_id

        # Title - may have newlines we need to clean
        title = get_text("title")
        title = " ".join(title.split())

        # Authors
        authors = []
        for author in entry.findall(f"{ATOM_NS}author"):
            name = author.find(f"{ATOM_NS}name")
            if name is not None and name.text:
                authors.append(name.text.strip())

        # Abstract (called "summary" in Atom)
        abstract = get_text("summary")
        abstract = " ".join(abstract.split())  # Normalize whitespace

        # Categories
        categories = []
        for cat in entry.findall(f"{ARXIV_NS}primary_category"):
            term = cat.get("term")
            if term:
                categories.append(term)
        for cat in entry.findall(f"{ATOM_NS}category"):
            term = cat.get("term")
            if term and term not in categories:
                categories.append(term)

        # Links
        pdf_url = ""
        abs_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
            elif link.get("type") == "text/html":
                abs_url = link.get("href", "")

        # Fallback PDF URL construction
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if not abs_url and arxiv_id:
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"

        # Optional fields from arxiv namespace
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
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        search_field: str = "all"
    ) -> list[ArxivPaper]:
        """
        Search arXiv for papers.

        Args:
            query: Search query string
            max_results: Maximum number of results (default 10)
            sort_by: "relevance", "lastUpdatedDate", or "submittedDate"
            sort_order: "ascending" or "descending"
            search_field: "all", "ti" (title), "au" (author), "abs" (abstract),
                         "cat" (category), "id" (arXiv ID)

        Returns:
            List of ArxivPaper objects
        """
        self._wait_for_rate_limit()

        # Build search query with field prefix if specified
        if search_field != "all":
            search_query = f"{search_field}:{query}"
        else:
            search_query = query

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
            return []

        # Parse Atom XML
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError:
            return []

        papers = []
        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                paper = self._parse_entry(entry)
                papers.append(paper)
            except Exception:
                continue

        return papers

    def get_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a specific paper by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "1706.03762" or "2301.00001")

        Returns:
            ArxivPaper or None if not found
        """
        # Clean up ID - remove version suffix if present
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        self._wait_for_rate_limit()

        params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }

        try:
            r = self.session.get(self.BASE_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
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
        """
        Extract arXiv ID from a URL or string.

        Handles formats like:
        - https://arxiv.org/abs/1706.03762
        - https://arxiv.org/pdf/1706.03762.pdf
        - arxiv:1706.03762
        - 1706.03762
        - 2301.00001v2

        Returns:
            arXiv ID without version, or None if not found
        """
        patterns = [
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",
            r"arxiv:(\d{4}\.\d{4,5})",
            r"\b(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def download_pdf_text(
        self,
        paper: ArxivPaper,
        full_text: bool = True,
        max_pages: Optional[int] = None
    ) -> Optional[str]:
        """
        Download PDF and extract text.

        Args:
            paper: ArxivPaper object with pdf_url
            full_text: If True, extract all pages; if False, just first page
            max_pages: Optional maximum number of pages to extract

        Returns:
            Extracted text or None if failed
        """
        if not paper.pdf_url:
            return None

        try:
            r = self.session.get(paper.pdf_url, timeout=60)
            r.raise_for_status()
        except requests.RequestException:
            return None

        # Verify we got a PDF
        if not r.content.startswith(b"%PDF-"):
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(r.content)
                tmp.flush()
                tmp_path = tmp.name

            # Build pdftotext command
            cmd = ["pdftotext"]
            if not full_text:
                cmd.extend(["-f", "1", "-l", "1"])
            elif max_pages:
                cmd.extend(["-f", "1", "-l", str(max_pages)])
            cmd.extend([tmp_path, "-"])

            text = subprocess.check_output(cmd, text=True, timeout=60).strip()

            os.unlink(tmp_path)
            return text if text else None

        except Exception:
            try:
                os.unlink(tmp_path)
            except:
                pass
            return None

    def search_recent(
        self,
        category: str,
        max_results: int = 10,
        days_back: int = 7
    ) -> list[ArxivPaper]:
        """
        Search for recent papers in a category.

        Args:
            category: arXiv category (e.g., "cs.CL", "cs.LG", "stat.ML")
            max_results: Maximum number of results
            days_back: How many days back to search

        Returns:
            List of ArxivPaper objects, sorted by submission date
        """
        return self.search(
            query=category,
            max_results=max_results,
            search_field="cat",
            sort_by="submittedDate",
            sort_order="descending"
        )


# Convenience function for direct use
def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    """
    Quick search function that returns dicts.

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        List of paper dicts
    """
    client = ArxivClient()
    papers = client.search(query, max_results=max_results)
    return [p.to_dict() for p in papers]


def arxiv_get(arxiv_id: str) -> Optional[dict]:
    """
    Quick fetch function for a single paper.

    Args:
        arxiv_id: arXiv ID (e.g., "1706.03762")

    Returns:
        Paper dict or None
    """
    client = ArxivClient()
    paper = client.get_paper(arxiv_id)
    return paper.to_dict() if paper else None
