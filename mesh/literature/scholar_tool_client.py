import json
from .scholar_pdf import ScholarSearch
from .semantic import SemanticScholar
from .sources.arxiv import ArxivClient
from .sources.pubmed import PubMedClient
from .literature_search import LiteratureSearch, Source, Paper
from .fulltext_extractor import FulltextExtractor, ExtractedText
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional

from ..paths import resolve_path

# class FulltextFetcher:
#     cookie_file = "/tmp/scholar_cookies.txt"                                                                             
#     def fetch(self, url):
#         # Use lynx to fetch the PDF through any cookie walls
#         with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:

#             try:
#                 pdf_bytes = subprocess.check_output(
#                     [
#                         "lynx",
#                         "-source",
#                         "-accept_all_cookies",
#                         f"-cookie_file={cookie_file}",
#                         f"-cookie_save_file={cookie_file}",
#                         "-useragent=Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
#                         url,
#                     ],
#                     stderr=subprocess.DEVNULL,
#                     timeout=20
#                 )
#             except Exception:
#                 return None

#             # Lynx returns HTML instead of PDF if it hits a hard block
#             # Quick MIME sniff: PDF always starts with %PDF-
#             if not pdf_bytes.startswith(b"%PDF-"):
#                 return None

#             tmp.write(pdf_bytes)
#             tmp.flush()

#             try:
#                 text = subprocess.check_output(
#                     ["pdftotext", tmp.name, "-"],
#                     text=True
#                 ).strip()
#                 return text if text else None
#             except Exception:
#                 return None

from bs4 import BeautifulSoup

class ScholarToolClient:
    """
    Handles tool calls for scholar_search, returning abstracts from ScholarSearch().
    """

    def __init__(self):
        #self.sch = ScholarSearch()
        self.sch = SemanticScholar()
        self.arxiv = ArxivClient(delay_seconds=1.0)
        self.pubmed = PubMedClient()
        self.cookie_file = "/tmp/scholar_cookies.txt"

        # Unified search interface
        self.literature = LiteratureSearch(
            enable_arxiv=True,
            enable_pubmed=True,
            enable_semantic_scholar=True,
            enable_google_scholar=False,  # Off by default due to rate limits
        )

    def _extract_text_from_file(self, path: Path) -> Optional[str]:
        """
        Given a local file path (PDF or HTML/text), return extracted text or None.
        """
        try:
            with path.open("rb") as f:
                header = f.read(5)
        except Exception:
            return None

        # ------------------------
        # Case 1: PDF
        # ------------------------
        if header.startswith(b"%PDF-"):
            try:
                text = subprocess.check_output(
                    ["pdftotext", str(path), "-"],
                    text=True,
                ).strip()
            except Exception:
                return None
            return text or None

        # ------------------------
        # Case 2: HTML / generic text
        # ------------------------
        try:
            with path.open("rb") as f:
                raw = f.read()
        except Exception:
            return None

        # Try HTML parsing first
        try:
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = "\n".join(
                line.strip() for line in text.splitlines() if line.strip()
            )
            if text:
                return text
        except Exception:
            # ignore and fall back to plain text decode
            pass

        # Fallback: just decode as UTF-8 text
        try:
            return raw.decode("utf-8", errors="replace") or None
        except Exception:
            return None

    def handle_fulltext(self, args: dict) -> Optional[str]:
        """
        Handler for fetch_fulltext_pdf tool.
        Args is a dict from the tool call, e.g. {"url": "http://...", ...}
        Supports both remote URLs and local file paths.
        """
        url_or_path = args.get("url") or args.get("path")
        if not isinstance(url_or_path, str) or not url_or_path.strip():
            return None
        url_or_path = url_or_path.strip()

        if url_or_path.startswith("file://"):
            # Strip scheme; for Unix paths file:///path → /path
            fs_path = url_or_path[len("file://"):]
            expanded = resolve_path(fs_path)
            local_path = Path(expanded)
            if local_path.is_file():
                return self._extract_text_from_file(local_path)
            # if it *still* isn't a file, just fall back to curl branch below
        
        # Local file?
        expanded = resolve_path(url_or_path)
        local_path = Path(expanded)
        if local_path.is_file():
            return self._extract_text_from_file(local_path)

        # Otherwise, treat as remote URL
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            try:
                subprocess.check_call(
                    [
                        "curl",
                        "-L",
                        "-A", "Mozilla/5.0",
                        "-b", self.cookie_file,
                        "-c", self.cookie_file,
                        "--silent",
                        "--show-error",
                        "--fail",
                        "-o", tmp.name,
                        url_or_path,
                    ],
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                return None

            return self._extract_text_from_file(Path(tmp.name))
        
    def fetch(self, url_or_path: str) -> Optional[str]:
        """
        Fetch PDF/HTML/text from either a remote URL or a local file path,
        and return extracted plain text, or None on failure.

        - If url_or_path looks like an existing local file, it is read directly.
        - Otherwise, it's treated as a URL and fetched via curl.
        """
        # First: check if this is an existing local file
        expanded = resolve_path(url_or_path)
        local_path = Path(expanded)
        if local_path.is_file():
            return self._extract_text_from_file(local_path)

        # Otherwise, treat as remote URL
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            try:
                subprocess.check_call(
                    [
                        "curl",
                        "-L",                    # follow redirects
                        "-A", "Mozilla/5.0",     # modern browser UA
                        "-b", self.cookie_file,  # load cookies
                        "-c", self.cookie_file,  # save cookies
                        "--silent",
                        "--show-error",
                        "--fail",                # fail on HTTP errors
                        "-o", tmp.name,
                        url_or_path,
                    ],
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                return None

            return self._extract_text_from_file(Path(tmp.name))
        
    def handle_scholar(self, args) :
        query = args.get("query", "").strip()
        #print( "HS query seen: " + query, file=sys.stderr )
        n = int(args.get("n_results", 5))

        if not query:
            return {
                "error": "Missing required parameter: query"
            }

        # Call ScholarSearch
        try:
            results = self.sch.search(query, n)
        except Exception as e:
            return {
                "error": f"scholar_search failed: {e}"
            }

        #print( "HS results: ", file=sys.stderr )
        #print( results, file=sys.stderr )
        # Compact JSON payload
        payload = []
        for r in results:
            payload.append({
                "authors": r.get("author"),
                "year": r.get("year"),
                "title": r["title"],
                "abstract": r.get("abstract"),
                "publisher_url": r.get("publisher_url"),
                "pdf_url": r.get("pdf_url"),
            })

        return payload

    def handle_arxiv_search(self, args: dict):
        """
        Handler for arxiv_search tool.

        Args:
            query: Search query string
            max_results: Maximum number of results (default 10)
            search_field: "all", "ti" (title), "au" (author), "abs" (abstract)
        """
        query = args.get("query", "").strip()
        if not query:
            return {"error": "Missing required parameter: query"}

        max_results = int(args.get("max_results", 10))
        search_field = args.get("search_field", "all")

        try:
            papers = self.arxiv.search(
                query=query,
                max_results=max_results,
                search_field=search_field
            )
        except Exception as e:
            return {"error": f"arxiv_search failed: {e}"}

        # Convert to compact payload
        payload = []
        for p in papers:
            payload.append({
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors[:5],  # Limit to first 5 authors
                "abstract": p.abstract,
                "published": p.published[:10] if p.published else None,  # Just date
                "categories": p.categories,
                "pdf_url": p.pdf_url,
                "abs_url": p.abs_url,
            })

        return payload

    def handle_arxiv_get(self, args: dict):
        """
        Handler for arxiv_get tool - fetch a specific paper by ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "1706.03762")
        """
        arxiv_id = args.get("arxiv_id", "").strip()
        if not arxiv_id:
            # Try to extract from URL if provided
            url = args.get("url", "").strip()
            if url:
                arxiv_id = self.arxiv.extract_arxiv_id(url)

        if not arxiv_id:
            return {"error": "Missing required parameter: arxiv_id or url"}

        try:
            paper = self.arxiv.get_paper(arxiv_id)
        except Exception as e:
            return {"error": f"arxiv_get failed: {e}"}

        if not paper:
            return {"error": f"Paper not found: {arxiv_id}"}

        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "published": paper.published[:10] if paper.published else None,
            "categories": paper.categories,
            "pdf_url": paper.pdf_url,
            "abs_url": paper.abs_url,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
        }

    def handle_arxiv_fulltext(self, args: dict):
        """
        Handler for arxiv_fulltext tool - download and extract full paper text.

        Args:
            arxiv_id: arXiv paper ID (e.g., "1706.03762")
            max_pages: Optional maximum number of pages to extract
        """
        arxiv_id = args.get("arxiv_id", "").strip()
        if not arxiv_id:
            url = args.get("url", "").strip()
            if url:
                arxiv_id = self.arxiv.extract_arxiv_id(url)

        if not arxiv_id:
            return {"error": "Missing required parameter: arxiv_id or url"}

        max_pages = args.get("max_pages")
        if max_pages:
            max_pages = int(max_pages)

        try:
            paper = self.arxiv.get_paper(arxiv_id)
            if not paper:
                return {"error": f"Paper not found: {arxiv_id}"}

            text = self.arxiv.download_pdf_text(
                paper,
                full_text=True,
                max_pages=max_pages
            )
        except Exception as e:
            return {"error": f"arxiv_fulltext failed: {e}"}

        if not text:
            return {"error": f"Could not extract text from paper: {arxiv_id}"}

        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "text": text,
            "char_count": len(text),
        }

    def handle_pubmed_search(self, args: dict):
        """
        Handler for pubmed_search tool.

        Args:
            query: Search query (supports PubMed syntax)
            max_results: Maximum number of results (default 10)
            sort: Sort order - "relevance", "pub_date", "first_author"
            min_date: Minimum publication date (YYYY or YYYY/MM/DD)
            max_date: Maximum publication date (YYYY or YYYY/MM/DD)
        """
        query = args.get("query", "").strip()
        if not query:
            return {"error": "Missing required parameter: query"}

        max_results = int(args.get("max_results", 10))
        sort = args.get("sort", "relevance")
        min_date = args.get("min_date")
        max_date = args.get("max_date")

        try:
            papers = self.pubmed.search(
                query=query,
                max_results=max_results,
                sort=sort,
                min_date=min_date,
                max_date=max_date,
            )
        except Exception as e:
            return {"error": f"pubmed_search failed: {e}"}

        # Convert to compact payload
        payload = []
        for p in papers:
            payload.append({
                "pmid": p.pmid,
                "title": p.title,
                "authors": p.authors[:5],  # Limit to first 5 authors
                "abstract": p.abstract[:1000] if p.abstract else None,  # Truncate long abstracts
                "journal": p.journal,
                "pub_date": p.pub_date,
                "doi": p.doi,
                "pmcid": p.pmcid,  # Important: indicates full text may be available
                "pubmed_url": p.pubmed_url,
                "pmc_url": p.pmc_url,
            })

        return payload

    def handle_pubmed_get(self, args: dict):
        """
        Handler for pubmed_get tool - fetch a specific paper by PMID.

        Args:
            pmid: PubMed ID (numeric string)
        """
        pmid = args.get("pmid", "").strip()
        if not pmid:
            return {"error": "Missing required parameter: pmid"}

        try:
            paper = self.pubmed.get_paper(pmid)
        except Exception as e:
            return {"error": f"pubmed_get failed: {e}"}

        if not paper:
            return {"error": f"Paper not found: {pmid}"}

        return {
            "pmid": paper.pmid,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "journal": paper.journal,
            "pub_date": paper.pub_date,
            "doi": paper.doi,
            "pmcid": paper.pmcid,
            "keywords": paper.keywords,
            "mesh_terms": paper.mesh_terms,
            "pubmed_url": paper.pubmed_url,
            "pmc_url": paper.pmc_url,
        }

    def handle_pubmed_fulltext(self, args: dict):
        """
        Handler for pubmed_fulltext tool - get full text from PMC.

        Only works for papers in the PMC Open Access Subset.

        Args:
            pmid: PubMed ID
            pmcid: PMC ID (e.g., "PMC1234567") - preferred if available
        """
        pmid = args.get("pmid", "").strip() if args.get("pmid") else None
        pmcid = args.get("pmcid", "").strip() if args.get("pmcid") else None

        if not pmid and not pmcid:
            return {"error": "Missing required parameter: pmid or pmcid"}

        try:
            text = self.pubmed.get_fulltext(pmid=pmid, pmcid=pmcid)
        except Exception as e:
            return {"error": f"pubmed_fulltext failed: {e}"}

        if not text:
            return {
                "error": f"Full text not available for this paper. "
                         f"It may not be in the PMC Open Access Subset."
            }

        # Get paper metadata for context
        paper = None
        if pmid:
            paper = self.pubmed.get_paper(pmid)

        result = {
            "text": text,
            "char_count": len(text),
        }

        if paper:
            result["pmid"] = paper.pmid
            result["title"] = paper.title
            result["pmcid"] = paper.pmcid

        return result

    def handle_pubmed_related(self, args: dict):
        """
        Handler for pubmed_related tool - find papers related to a given paper.

        Args:
            pmid: PubMed ID of the source paper
            max_results: Maximum number of related papers (default 10)
        """
        pmid = args.get("pmid", "").strip()
        if not pmid:
            return {"error": "Missing required parameter: pmid"}

        max_results = int(args.get("max_results", 10))

        try:
            papers = self.pubmed.get_related_papers(pmid, max_results=max_results)
        except Exception as e:
            return {"error": f"pubmed_related failed: {e}"}

        if not papers:
            return {"message": "No related papers found", "results": []}

        # Convert to compact payload
        payload = []
        for p in papers:
            payload.append({
                "pmid": p.pmid,
                "title": p.title,
                "authors": p.authors[:3],  # First 3 authors
                "abstract": p.abstract[:500] if p.abstract else None,
                "journal": p.journal,
                "pub_date": p.pub_date,
                "pmcid": p.pmcid,
                "pubmed_url": p.pubmed_url,
            })

        return payload

    # =========================================================================
    # Unified Literature Search Handlers
    # =========================================================================

    def handle_literature_search(self, args: dict):
        """
        Handler for literature_search tool - unified search across all sources.

        Automatically routes to the best sources based on query domain:
        - CS/ML queries → arXiv, Semantic Scholar
        - Biomedical queries → PubMed, Semantic Scholar
        - General queries → All sources

        Args:
            query: Search query string (required)
            max_results: Maximum number of results (default 10)
            sources: Optional list of sources to use ("arxiv", "pubmed", "semantic_scholar")
        """
        query = args.get("query", "").strip()
        if not query:
            return {"error": "Missing required parameter: query"}

        max_results = int(args.get("max_results", 10))

        # Parse sources if specified
        source_list = None
        sources_arg = args.get("sources")
        if sources_arg:
            source_map = {
                "arxiv": Source.ARXIV,
                "pubmed": Source.PUBMED,
                "semantic_scholar": Source.SEMANTIC_SCHOLAR,
                "google_scholar": Source.GOOGLE_SCHOLAR,
            }
            if isinstance(sources_arg, list):
                source_list = [source_map[s] for s in sources_arg if s in source_map]
            elif isinstance(sources_arg, str):
                source_list = [source_map[sources_arg]] if sources_arg in source_map else None

        try:
            papers = self.literature.search(
                query=query,
                max_results=max_results,
                sources=source_list,
            )
        except Exception as e:
            return {"error": f"literature_search failed: {e}"}

        # Convert to compact payload
        payload = []
        for p in papers:
            payload.append({
                "title": p.title,
                "authors": p.authors[:5],  # Limit to first 5 authors
                "abstract": p.abstract[:800] if p.abstract else None,  # Truncate long abstracts
                "year": p.year,
                "source": p.source.value if p.source else None,
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
                "pmid": p.pmid,
                "pmcid": p.pmcid,
                "pdf_url": p.pdf_url,
                "publisher_url": p.publisher_url,
            })

        return {
            "query": query,
            "count": len(payload),
            "results": payload,
        }

    def handle_literature_fulltext(self, args: dict):
        """
        Handler for literature_fulltext tool - get full text for a paper.

        Tries multiple sources in cascade:
        1. arXiv PDF (if arxiv_id present)
        2. PMC full text (if pmcid present)
        3. Publisher HTML
        4. Abstract fallback

        Args:
            arxiv_id: arXiv paper ID (e.g., "1706.03762")
            pmid: PubMed ID
            doi: Digital Object Identifier
            max_chars: Optional character limit (default: no limit)
            max_tokens: Optional token limit (overrides max_chars)
            max_pages: Limit PDF extraction to first N pages
        """
        arxiv_id = args.get("arxiv_id", "").strip() if args.get("arxiv_id") else None
        pmid = args.get("pmid", "").strip() if args.get("pmid") else None
        doi = args.get("doi", "").strip() if args.get("doi") else None
        max_chars = args.get("max_chars")
        max_tokens = args.get("max_tokens")
        max_pages = args.get("max_pages")

        if not any([arxiv_id, pmid, doi]):
            return {"error": "Missing required parameter: arxiv_id, pmid, or doi"}

        if max_chars:
            max_chars = int(max_chars)
        if max_tokens:
            max_tokens = int(max_tokens)
        if max_pages:
            max_pages = int(max_pages)

        try:
            paper = self.literature.get_paper_by_id(
                arxiv_id=arxiv_id,
                pmid=pmid,
                doi=doi,
            )
        except Exception as e:
            return {"error": f"Failed to fetch paper: {e}"}

        if not paper:
            return {"error": "Paper not found with provided identifier(s)"}

        try:
            text = self.literature.get_fulltext(
                paper,
                max_chars=max_chars,
                max_tokens=max_tokens,
                max_pages=max_pages,
            )
        except Exception as e:
            return {"error": f"Failed to get full text: {e}"}

        if not text:
            return {"error": "Full text not available for this paper"}

        return {
            "title": paper.title,
            "arxiv_id": paper.arxiv_id,
            "pmid": paper.pmid,
            "doi": paper.doi,
            "text": text,
            "char_count": len(text),
            "estimated_tokens": len(text) // 4,
        }

    def handle_literature_sections(self, args: dict):
        """
        Handler for literature_sections tool - extract specific sections from a paper.

        Extracts named sections like Abstract, Introduction, Methods, Results, etc.

        Args:
            arxiv_id: arXiv paper ID (e.g., "1706.03762")
            pmid: PubMed ID
            doi: Digital Object Identifier
            sections: List of section names to extract (default: all standard sections)
            max_pages: Limit PDF extraction to first N pages
        """
        arxiv_id = args.get("arxiv_id", "").strip() if args.get("arxiv_id") else None
        pmid = args.get("pmid", "").strip() if args.get("pmid") else None
        doi = args.get("doi", "").strip() if args.get("doi") else None
        sections_arg = args.get("sections", ["abstract", "introduction", "methods", "results", "conclusion"])
        max_pages = args.get("max_pages")

        if not any([arxiv_id, pmid, doi]):
            return {"error": "Missing required parameter: arxiv_id, pmid, or doi"}

        if max_pages:
            max_pages = int(max_pages)

        # Parse sections list
        if isinstance(sections_arg, str):
            sections_list = [s.strip().lower() for s in sections_arg.split(",")]
        elif isinstance(sections_arg, list):
            sections_list = [s.strip().lower() for s in sections_arg]
        else:
            sections_list = ["abstract", "introduction", "methods", "results", "conclusion"]

        try:
            paper = self.literature.get_paper_by_id(
                arxiv_id=arxiv_id,
                pmid=pmid,
                doi=doi,
            )
        except Exception as e:
            return {"error": f"Failed to fetch paper: {e}"}

        if not paper:
            return {"error": "Paper not found with provided identifier(s)"}

        try:
            extracted = self.literature.get_fulltext_with_sections(
                paper,
                max_pages=max_pages,
            )
        except Exception as e:
            return {"error": f"Failed to extract sections: {e}"}

        if not extracted:
            return {"error": "Could not extract text from this paper"}

        # Filter to requested sections
        found_sections = {
            s: extracted.sections.get(s, "")
            for s in sections_list
            if extracted.sections.get(s)
        }

        return {
            "title": paper.title,
            "arxiv_id": paper.arxiv_id,
            "pmid": paper.pmid,
            "doi": paper.doi,
            "source_type": extracted.source_type,
            "total_char_count": extracted.char_count,
            "estimated_tokens": extracted.estimated_tokens,
            "truncated": extracted.truncated,
            "sections_found": list(found_sections.keys()),
            "sections": found_sections,
        }

    def handle_extract_url(self, args: dict):
        """
        Handler for extract_url tool - extract full text from any PDF or HTML URL.

        Works with arXiv, publisher sites, local files, etc.

        Args:
            url: URL to PDF or HTML page
            max_chars: Optional character limit (default: 200k)
            max_tokens: Optional token limit (overrides max_chars)
            max_pages: Limit PDF extraction to first N pages
            extract_sections: Whether to identify sections (default: true)
        """
        url = args.get("url", "").strip()
        if not url:
            return {"error": "Missing required parameter: url"}

        max_chars = args.get("max_chars")
        max_tokens = args.get("max_tokens")
        max_pages = args.get("max_pages")
        extract_sections = args.get("extract_sections", True)

        if max_chars:
            max_chars = int(max_chars)
        if max_tokens:
            max_tokens = int(max_tokens)
        if max_pages:
            max_pages = int(max_pages)

        extractor = FulltextExtractor(
            max_chars=max_chars or 200_000,
            max_tokens=max_tokens,
        )

        try:
            result = extractor.extract_from_pdf_url(
                url,
                max_pages=max_pages,
                extract_sections=extract_sections,
            )
        except Exception as e:
            return {"error": f"PDF extraction failed: {e}"}

        if not result:
            # Try HTML extraction
            try:
                r = extractor.session.get(url, timeout=30)
                r.raise_for_status()
                result = extractor.extract_from_html(r.text, extract_sections=extract_sections)
            except Exception as e:
                return {"error": f"Extraction failed: {e}"}

        if not result:
            return {"error": "Could not extract text from URL"}

        response = {
            "url": url,
            "source_type": result.source_type,
            "char_count": result.char_count,
            "estimated_tokens": result.estimated_tokens,
            "truncated": result.truncated,
            "text": result.text,
        }

        if result.original_char_count:
            response["original_char_count"] = result.original_char_count

        if result.sections:
            response["sections_found"] = list(result.sections.keys())

        return response

    def dispatch_tool_call(self, tool_call):
        """
        Handle one tool call.
        Returns: (tool_message_dict, is_error)
        """

        # Attempt both OpenAI and dict-style tool-call inputs
        try:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments
            call_id = tool_call.id
            print( f"Tool call:name={name}, args={raw_args}" )
        except AttributeError:
            try:
                name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                call_id = tool_call.get("id", "")
            except Exception:
                return ({
                    "role": "tool",
                    "tool_call_id": "",
                    "content": "Malformed tool call: missing fields."
                }, True)

        # Parse args
        try:
            args = json.loads(raw_args) if raw_args else {}
        except Exception as e:
            return ({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Invalid JSON in tool arguments: {e}"
            }, True)
            
        handlers = {
            "scholar_search": self.handle_scholar,
            "fetch_fulltext_pdf": self.handle_fulltext,
            "arxiv_search": self.handle_arxiv_search,
            "arxiv_get": self.handle_arxiv_get,
            "arxiv_fulltext": self.handle_arxiv_fulltext,
            "pubmed_search": self.handle_pubmed_search,
            "pubmed_get": self.handle_pubmed_get,
            "pubmed_fulltext": self.handle_pubmed_fulltext,
            "pubmed_related": self.handle_pubmed_related,
            # Unified interface
            "literature_search": self.handle_literature_search,
            "literature_fulltext": self.handle_literature_fulltext,
            # Section extraction and URL tools
            "literature_sections": self.handle_literature_sections,
            "extract_url": self.handle_extract_url,
        }

        if name in handlers:
            payload = handlers[name](args)

            return ({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(payload, ensure_ascii=False, indent=2)
            }, False)
        else:
            return ({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Unknown tool: {name}"
            }, True)

        

