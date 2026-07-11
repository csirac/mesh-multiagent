"""
Unified Literature Search Interface.

Provides a single interface for searching across multiple academic sources:
- arXiv (CS, ML, Physics, Math)
- PubMed/PMC (Biomedical)
- Semantic Scholar (General)
- Google Scholar (General, fallback)

Features:
- Smart source routing based on query/domain
- Result deduplication by DOI/title
- Full-text cascade (arXiv PDF → PMC XML → publisher HTML → abstract)
- Graceful degradation when sources fail
- Per-source caching with TTL
- Retry with exponential backoff
- Circuit breaker for failing sources
"""

import re
import difflib
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum

from .sources.arxiv import ArxivClient, ArxivPaper
from .sources.pubmed import PubMedClient, PubMedPaper
from .sources.cache import LiteratureCache
from .sources.retry import (
    RetryConfig, CircuitBreaker, BlockedError,
    with_retry, RateLimitError,
)
from .semantic import SemanticScholar
from .scholar_pdf import ScholarSearch
from .fulltext_extractor import FulltextExtractor, ExtractedText

logger = logging.getLogger(__name__)


class Source(Enum):
    """Academic literature sources."""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    GOOGLE_SCHOLAR = "google_scholar"


@dataclass
class Paper:
    """
    Unified paper representation across all sources.

    Normalizes metadata from arXiv, PubMed, Semantic Scholar, and Google Scholar
    into a single consistent format.
    """
    title: str
    authors: list[str]
    abstract: str

    # Identifiers
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None

    # Metadata
    year: Optional[int] = None
    pub_date: Optional[str] = None
    journal: Optional[str] = None
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # URLs
    pdf_url: Optional[str] = None
    publisher_url: Optional[str] = None

    # Source tracking
    source: Optional[Source] = None

    # Full text (populated on demand)
    full_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "year": self.year,
            "pub_date": self.pub_date,
            "journal": self.journal,
            "categories": self.categories,
            "keywords": self.keywords,
            "pdf_url": self.pdf_url,
            "publisher_url": self.publisher_url,
            "source": self.source.value if self.source else None,
        }

    @property
    def normalized_title(self) -> str:
        """Normalized title for deduplication."""
        if not self.title:
            return ""
        t = ''.join(
            c for c in unicodedata.normalize('NFD', self.title)
            if unicodedata.category(c) != 'Mn'
        )
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip().lower()


class LiteratureSearch:
    """
    Unified interface for searching academic literature across multiple sources.

    Features:
    - Smart routing: Automatically selects best sources based on query domain
    - Deduplication: Merges results by DOI or title similarity
    - Full-text cascade: Tries multiple sources to get complete paper text
    - Graceful degradation: Returns partial results if some sources fail

    Example:
        search = LiteratureSearch()

        # Auto-routes to arXiv for ML paper
        papers = search.search("transformer attention mechanism")

        # Force specific source
        papers = search.search("COVID-19 vaccine", sources=[Source.PUBMED])

        # Get full text
        text = search.get_fulltext(papers[0])
    """

    # Domain keywords for source routing
    CS_ML_KEYWORDS = {
        "neural", "network", "deep learning", "machine learning", "transformer",
        "attention", "llm", "language model", "gpt", "bert", "diffusion",
        "reinforcement learning", "computer vision", "nlp", "algorithm",
        "optimization", "convolution", "recurrent", "gan", "autoencoder",
        "embedding", "classification", "regression", "clustering", "ai",
        "artificial intelligence", "robotics", "autonomous", "cuda", "gpu",
        "pytorch", "tensorflow", "backpropagation", "gradient descent"
    }

    BIO_KEYWORDS = {
        "gene", "protein", "cell", "disease", "cancer", "tumor", "clinical",
        "patient", "treatment", "therapy", "drug", "vaccine", "covid",
        "coronavirus", "virus", "bacteria", "infection", "immune", "genomic",
        "dna", "rna", "mrna", "mutation", "biomarker", "pathogen", "epidemic",
        "pandemic", "syndrome", "symptom", "diagnosis", "prognosis", "trial"
    }

    # Default timeouts per source (in seconds)
    DEFAULT_TIMEOUTS = {
        Source.ARXIV: 30,
        Source.PUBMED: 30,
        Source.SEMANTIC_SCHOLAR: 15,
        Source.GOOGLE_SCHOLAR: 20,
    }

    def __init__(
        self,
        enable_arxiv: bool = True,
        enable_pubmed: bool = True,
        enable_semantic_scholar: bool = True,
        enable_google_scholar: bool = True,  # Enabled by default; failures handled gracefully
        s2_api_key: Optional[str] = None,
        # Robustness options
        enable_caching: bool = True,
        cache_dir: str = "~/.cache/literature_search",
        retry_config: Optional[RetryConfig] = None,
        custom_timeouts: Optional[dict[Source, int]] = None,
    ):
        """
        Initialize LiteratureSearch with configurable sources.

        Args:
            enable_arxiv: Enable arXiv API
            enable_pubmed: Enable PubMed/PMC APIs
            enable_semantic_scholar: Enable Semantic Scholar API
            enable_google_scholar: Enable Google Scholar scraping (may hit CAPTCHAs)
            s2_api_key: Semantic Scholar API key (optional, for higher rate limits)
            enable_caching: Enable result caching (recommended)
            cache_dir: Directory for disk cache
            retry_config: Custom retry configuration
            custom_timeouts: Custom timeouts per source
        """
        self.arxiv = ArxivClient(delay_seconds=1.0) if enable_arxiv else None
        self.pubmed = PubMedClient() if enable_pubmed else None
        self.semantic_scholar = SemanticScholar() if enable_semantic_scholar else None
        self.google_scholar = ScholarSearch(s2_api_key=s2_api_key) if enable_google_scholar else None

        self._enabled_sources = {
            Source.ARXIV: enable_arxiv,
            Source.PUBMED: enable_pubmed,
            Source.SEMANTIC_SCHOLAR: enable_semantic_scholar,
            Source.GOOGLE_SCHOLAR: enable_google_scholar,
        }

        # Timeouts per source
        self._timeouts = {**self.DEFAULT_TIMEOUTS, **(custom_timeouts or {})}

        # Retry configuration
        self._retry_config = retry_config or RetryConfig(
            max_retries=2,
            base_delay=1.0,
            max_delay=10.0,
        )

        # Circuit breakers per source (opens after 3 failures, recovers after 60s)
        self._circuit_breakers: dict[Source, CircuitBreaker] = {
            s: CircuitBreaker(failure_threshold=3, recovery_time=60.0)
            for s in Source
        }

        # Caching
        self._cache: Optional[LiteratureCache] = None
        if enable_caching:
            self._cache = LiteratureCache(
                use_disk_cache=True,
                disk_cache_dir=cache_dir,
            )

        # Legacy failure tracking (kept for get_source_status compatibility)
        self._source_failures: dict[Source, int] = {s: 0 for s in Source}
        self._max_consecutive_failures = 3

    def _detect_domain(self, query: str) -> list[Source]:
        """
        Detect query domain and return ordered list of sources to try.

        Returns sources ordered by relevance for the detected domain.
        """
        query_lower = query.lower()

        # Check for CS/ML keywords
        cs_score = sum(1 for kw in self.CS_ML_KEYWORDS if kw in query_lower)

        # Check for biomedical keywords
        bio_score = sum(1 for kw in self.BIO_KEYWORDS if kw in query_lower)

        # Determine primary domain
        if cs_score > bio_score and cs_score >= 1:
            # CS/ML domain - prioritize arXiv
            sources = [Source.ARXIV, Source.SEMANTIC_SCHOLAR, Source.GOOGLE_SCHOLAR]
        elif bio_score > cs_score and bio_score >= 1:
            # Biomedical domain - prioritize PubMed
            sources = [Source.PUBMED, Source.SEMANTIC_SCHOLAR, Source.GOOGLE_SCHOLAR]
        else:
            # General/unknown domain
            sources = [Source.SEMANTIC_SCHOLAR, Source.ARXIV, Source.PUBMED, Source.GOOGLE_SCHOLAR]

        # Filter to only enabled sources
        return [s for s in sources if self._enabled_sources.get(s, False)]

    def _convert_arxiv_paper(self, paper: ArxivPaper) -> Paper:
        """Convert ArxivPaper to unified Paper."""
        year = None
        if paper.published:
            match = re.match(r"(\d{4})", paper.published)
            if match:
                year = int(match.group(1))

        return Paper(
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            year=year,
            pub_date=paper.published[:10] if paper.published else None,
            journal=paper.journal_ref,
            categories=paper.categories,
            pdf_url=paper.pdf_url,
            publisher_url=paper.abs_url,
            source=Source.ARXIV,
        )

    def _convert_pubmed_paper(self, paper: PubMedPaper) -> Paper:
        """Convert PubMedPaper to unified Paper."""
        year = None
        if paper.pub_date:
            match = re.match(r"(\d{4})", paper.pub_date)
            if match:
                year = int(match.group(1))

        return Paper(
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            doi=paper.doi,
            pmid=paper.pmid,
            pmcid=paper.pmcid,
            year=year,
            pub_date=paper.pub_date,
            journal=paper.journal,
            keywords=paper.keywords + paper.mesh_terms,
            publisher_url=paper.pubmed_url,
            pdf_url=paper.pmc_url,  # PMC URL as "pdf" since it has full text
            source=Source.PUBMED,
        )

    def _convert_semantic_scholar_result(self, result: dict) -> Paper:
        """Convert Semantic Scholar result to unified Paper."""
        authors = result.get("author", [])
        if isinstance(authors, list) and authors and isinstance(authors[0], str):
            # Already a list of strings
            pass
        elif isinstance(authors, list):
            # List of dicts with name field
            authors = [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in authors]
        else:
            authors = []

        return Paper(
            title=result.get("title", ""),
            authors=authors,
            abstract=result.get("abstract", ""),
            year=result.get("year"),
            pdf_url=result.get("pdf_url"),
            publisher_url=result.get("publisher_url"),
            source=Source.SEMANTIC_SCHOLAR,
        )

    def _convert_google_scholar_result(self, result: dict) -> Paper:
        """Convert Google Scholar result to unified Paper."""
        # Parse authors and year from "authors_year" field
        authors_year = result.get("authors_year", "")
        authors = []
        year = None

        if authors_year:
            # Format: "A Author, B Author - Journal, 2023"
            parts = authors_year.split(" - ")
            if parts:
                authors = [a.strip() for a in parts[0].split(",") if a.strip()]
            # Try to extract year
            match = re.search(r"\b(19|20)\d{2}\b", authors_year)
            if match:
                year = int(match.group())

        return Paper(
            title=result.get("title", ""),
            authors=authors,
            abstract=result.get("abstract", "") or result.get("snippet", ""),
            year=year,
            pdf_url=result.get("pdf_url"),
            publisher_url=result.get("publisher_url"),
            source=Source.GOOGLE_SCHOLAR,
        )

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """
        Remove duplicate papers based on DOI or title similarity.

        Keeps the paper with more metadata when duplicates are found.
        """
        if not papers:
            return papers

        seen_dois: set[str] = set()
        seen_titles: dict[str, Paper] = {}  # normalized_title -> Paper
        deduplicated: list[Paper] = []

        for paper in papers:
            # Check DOI first (exact match)
            if paper.doi:
                doi_lower = paper.doi.lower()
                if doi_lower in seen_dois:
                    continue
                seen_dois.add(doi_lower)

            # Check title similarity
            norm_title = paper.normalized_title
            if not norm_title:
                deduplicated.append(paper)
                continue

            # Look for similar titles
            is_duplicate = False
            for existing_title, existing_paper in seen_titles.items():
                similarity = difflib.SequenceMatcher(None, norm_title, existing_title).ratio()
                if similarity > 0.85:  # High threshold for title matching
                    # Keep the one with more metadata
                    existing_score = sum([
                        bool(existing_paper.doi),
                        bool(existing_paper.arxiv_id),
                        bool(existing_paper.pmid),
                        len(existing_paper.abstract) > 100,
                    ])
                    new_score = sum([
                        bool(paper.doi),
                        bool(paper.arxiv_id),
                        bool(paper.pmid),
                        len(paper.abstract) > 100,
                    ])

                    if new_score > existing_score:
                        # Replace existing with new
                        deduplicated.remove(existing_paper)
                        seen_titles[norm_title] = paper
                        deduplicated.append(paper)

                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_titles[norm_title] = paper
                deduplicated.append(paper)

        return deduplicated

    def _is_source_available(self, source: Source) -> bool:
        """Check if a source is available (enabled and circuit not open)."""
        if not self._enabled_sources.get(source, False):
            return False
        # Use circuit breaker instead of simple failure count
        breaker = self._circuit_breakers.get(source)
        if breaker and not breaker.allow_request():
            return False
        return True

    def _record_success(self, source: Source) -> None:
        """Record a successful query, resetting circuit breaker."""
        self._source_failures[source] = 0
        breaker = self._circuit_breakers.get(source)
        if breaker:
            breaker.record_success()

    def _record_failure(self, source: Source, error: Optional[Exception] = None) -> None:
        """Record a failed query."""
        self._source_failures[source] += 1
        breaker = self._circuit_breakers.get(source)
        if breaker:
            breaker.record_failure()

        # Log the failure for debugging
        error_msg = str(error) if error else "Unknown error"
        logger.warning(f"Source {source.value} failed: {error_msg}")

    def _is_blocked_error(self, error: Exception, source: Source) -> bool:
        """Check if an error indicates the source is blocking us."""
        error_str = str(error).lower()

        # CAPTCHA detection
        if "captcha" in error_str or "unusual traffic" in error_str:
            return True

        # Rate limit (429) is handled by retry, but persistent 429 = blocked
        if "429" in error_str and self._source_failures[source] >= 2:
            return True

        # IP ban indicators
        if "403" in error_str or "forbidden" in error_str:
            return True

        # Google Scholar specific blocks
        if source == Source.GOOGLE_SCHOLAR:
            if "sorry" in error_str or "automated" in error_str:
                return True

        return False

    def reset_source(self, source: Source) -> None:
        """Manually reset a source's circuit breaker to re-enable it."""
        self._source_failures[source] = 0
        breaker = self._circuit_breakers.get(source)
        if breaker:
            breaker.reset()

    def get_source_status(self) -> dict[str, dict]:
        """Get status of all sources including circuit breaker state."""
        result = {}
        for s in Source:
            breaker = self._circuit_breakers.get(s)
            result[s.value] = {
                "enabled": self._enabled_sources.get(s, False),
                "failures": self._source_failures[s],
                "available": self._is_source_available(s),
                "circuit_state": breaker.state.value if breaker else "unknown",
            }
        return result

    def get_cache_stats(self) -> Optional[dict]:
        """Get cache statistics if caching is enabled."""
        if self._cache:
            return self._cache.get_stats()
        return None

    def search(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[list[Source]] = None,
        deduplicate: bool = True,
        use_cache: bool = True,
    ) -> list[Paper]:
        """
        Search for papers across configured sources.

        Args:
            query: Search query string
            max_results: Maximum number of results per source
            sources: Specific sources to search (auto-detected if None)
            deduplicate: Remove duplicate papers across sources
            use_cache: Whether to use cached results (default True)

        Returns:
            List of Paper objects, sorted by relevance

        Notes:
            Sources that fail repeatedly (3+ consecutive failures) are
            automatically skipped until reset via circuit breaker.
            Results are cached per-source with TTL for faster repeated queries.
        """
        if sources is None:
            sources = self._detect_domain(query)

        all_papers: list[Paper] = []
        errors: list[str] = []

        for source in sources:
            # Skip sources that have failed repeatedly (circuit breaker open)
            if not self._is_source_available(source):
                logger.debug(f"Skipping {source.value}: circuit breaker open")
                continue

            # Check cache first
            if use_cache and self._cache:
                cached = self._cache.get_search(source.value, query, max_results)
                if cached:
                    logger.debug(f"Cache hit for {source.value}: {query}")
                    papers_from_source = [self._dict_to_paper(d) for d in cached]
                    all_papers.extend(papers_from_source)
                    continue

            try:
                papers_from_source: list[Paper] = []

                if source == Source.ARXIV and self.arxiv:
                    arxiv_papers = self.arxiv.search(query, max_results=max_results)
                    papers_from_source = [self._convert_arxiv_paper(p) for p in arxiv_papers]

                elif source == Source.PUBMED and self.pubmed:
                    pubmed_papers = self.pubmed.search(query, max_results=max_results)
                    papers_from_source = [self._convert_pubmed_paper(p) for p in pubmed_papers]

                elif source == Source.SEMANTIC_SCHOLAR and self.semantic_scholar:
                    s2_results = self.semantic_scholar.search(query, nresults=max_results)
                    papers_from_source = [self._convert_semantic_scholar_result(r) for r in s2_results]

                elif source == Source.GOOGLE_SCHOLAR and self.google_scholar:
                    gs_results = self.google_scholar.search(query, nresults=max_results)
                    # Check for CAPTCHA/block in results
                    if not gs_results:
                        # Empty results might indicate a block
                        logger.debug(f"Empty results from Google Scholar for: {query}")
                    papers_from_source = [self._convert_google_scholar_result(r) for r in gs_results]

                # Record success and cache results
                if papers_from_source:
                    self._record_success(source)
                    all_papers.extend(papers_from_source)

                    # Cache the results
                    if self._cache:
                        cache_data = [p.to_dict() for p in papers_from_source]
                        self._cache.set_search(source.value, query, cache_data, max_results)

            except Exception as e:
                # Check if this is a blocking error (CAPTCHA, IP ban)
                if self._is_blocked_error(e, source):
                    logger.warning(f"Source {source.value} appears to be blocking requests")
                    # Record multiple failures to trigger circuit breaker faster
                    for _ in range(3):
                        self._record_failure(source, e)
                else:
                    self._record_failure(source, e)
                errors.append(f"{source.value}: {e}")
                continue

        if deduplicate:
            all_papers = self._deduplicate(all_papers)

        return all_papers[:max_results * 2]  # Return up to 2x max_results after dedup

    def _dict_to_paper(self, d: dict) -> Paper:
        """Convert a cached dict back to a Paper object."""
        source = None
        if d.get("source"):
            try:
                source = Source(d["source"])
            except ValueError:
                pass

        return Paper(
            title=d.get("title", ""),
            authors=d.get("authors", []),
            abstract=d.get("abstract", ""),
            doi=d.get("doi"),
            arxiv_id=d.get("arxiv_id"),
            pmid=d.get("pmid"),
            pmcid=d.get("pmcid"),
            year=d.get("year"),
            pub_date=d.get("pub_date"),
            journal=d.get("journal"),
            categories=d.get("categories", []),
            keywords=d.get("keywords", []),
            pdf_url=d.get("pdf_url"),
            publisher_url=d.get("publisher_url"),
            source=source,
        )

    def search_arxiv(self, query: str, max_results: int = 10) -> list[Paper]:
        """Convenience method to search only arXiv."""
        return self.search(query, max_results, sources=[Source.ARXIV])

    def search_pubmed(self, query: str, max_results: int = 10) -> list[Paper]:
        """Convenience method to search only PubMed."""
        return self.search(query, max_results, sources=[Source.PUBMED])

    def get_fulltext(
        self,
        paper: Paper,
        max_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_pages: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        Get full text for a paper, trying multiple sources.

        Cascade order:
        1. arXiv PDF (if arxiv_id present)
        2. PMC full text (if pmcid present)
        3. PDF URL via FulltextExtractor
        4. Publisher HTML abstract
        5. Return abstract as fallback

        Args:
            paper: Paper object to get full text for
            max_chars: Optional character limit for returned text
            max_tokens: Optional token limit (overrides max_chars)
            max_pages: Limit PDF extraction to first N pages
            use_cache: Whether to use cached fulltext (default True)

        Returns:
            Full text string or None if unavailable
        """
        # Generate a cache key based on paper identifiers
        cache_key = paper.arxiv_id or paper.pmid or paper.doi or paper.title[:50]

        # Check cache first
        if use_cache and self._cache and cache_key:
            cached = self._cache.get_fulltext(cache_key)
            if cached:
                logger.debug(f"Fulltext cache hit for: {cache_key}")
                return cached

        text = None
        extractor = FulltextExtractor(max_chars=max_chars or 200_000, max_tokens=max_tokens)

        # Try arXiv PDF first (using improved extractor)
        if paper.arxiv_id and self.arxiv:
            try:
                arxiv_paper = self.arxiv.get_paper(paper.arxiv_id)
                if arxiv_paper and arxiv_paper.pdf_url:
                    result = extractor.extract_from_pdf_url(
                        arxiv_paper.pdf_url,
                        max_pages=max_pages,
                    )
                    if result:
                        text = result.text
            except Exception as e:
                logger.debug(f"arXiv fulltext extraction failed: {e}")

        # Try PMC full text (structured XML, high quality)
        if not text and paper.pmcid and self.pubmed:
            try:
                text = self.pubmed.get_fulltext(pmcid=paper.pmcid)
            except Exception as e:
                logger.debug(f"PMC fulltext extraction failed: {e}")

        # Try PMID for PMC
        if not text and paper.pmid and self.pubmed:
            try:
                text = self.pubmed.get_fulltext(pmid=paper.pmid)
            except Exception as e:
                logger.debug(f"PMID fulltext extraction failed: {e}")

        # Try PDF URL directly with improved extractor
        if not text and paper.pdf_url:
            try:
                result = extractor.extract_from_pdf_url(
                    paper.pdf_url,
                    max_pages=max_pages,
                )
                if result:
                    text = result.text
            except Exception as e:
                logger.debug(f"PDF URL extraction failed: {e}")

        # Try publisher HTML
        if not text and paper.publisher_url and self.google_scholar:
            try:
                text = self.google_scholar.extract_publisher_abstract(paper.publisher_url)
            except Exception as e:
                logger.debug(f"Publisher HTML extraction failed: {e}")

        # Fallback to abstract
        if not text:
            text = paper.abstract

        # Cache the result if we got substantial text
        if text and len(text) > 500 and self._cache and cache_key:
            self._cache.set_fulltext(cache_key, text)

        return text

    def get_fulltext_with_sections(
        self,
        paper: Paper,
        max_chars: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> Optional[ExtractedText]:
        """
        Get full text with section extraction.

        Returns ExtractedText object with:
        - text: Full extracted text
        - sections: Dict of section_name -> content
        - char_count, estimated_tokens, truncated flag

        Args:
            paper: Paper object to extract from
            max_chars: Maximum characters (default 200k)
            max_pages: Limit PDF extraction to first N pages

        Returns:
            ExtractedText object or None if extraction failed
        """
        extractor = FulltextExtractor(max_chars=max_chars or 200_000)

        # Try arXiv PDF
        if paper.arxiv_id and self.arxiv:
            try:
                arxiv_paper = self.arxiv.get_paper(paper.arxiv_id)
                if arxiv_paper and arxiv_paper.pdf_url:
                    result = extractor.extract_from_pdf_url(
                        arxiv_paper.pdf_url,
                        max_pages=max_pages,
                        extract_sections=True,
                    )
                    if result:
                        return result
            except Exception:
                pass

        # Try PDF URL directly
        if paper.pdf_url:
            try:
                result = extractor.extract_from_pdf_url(
                    paper.pdf_url,
                    max_pages=max_pages,
                    extract_sections=True,
                )
                if result:
                    return result
            except Exception:
                pass

        # Fallback: create ExtractedText from abstract
        if paper.abstract:
            from fulltext_extractor import ExtractedText
            return ExtractedText(
                text=paper.abstract,
                source_type="abstract",
                char_count=len(paper.abstract),
                estimated_tokens=len(paper.abstract) // 4,
                sections={"abstract": paper.abstract},
            )

        return None

    def get_paper_sections(
        self,
        paper: Paper,
        sections: list[str] = ["abstract", "introduction", "methods", "results", "conclusion"],
    ) -> dict[str, str]:
        """
        Extract specific sections from a paper.

        Args:
            paper: Paper to extract from
            sections: List of section names to extract

        Returns:
            Dict of section_name -> content (only found sections included)
        """
        result = self.get_fulltext_with_sections(paper)
        if not result:
            return {}

        return {s: result.sections.get(s, "") for s in sections if result.sections.get(s)}

    def get_paper_by_id(
        self,
        arxiv_id: Optional[str] = None,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
    ) -> Optional[Paper]:
        """
        Fetch a specific paper by identifier.

        Args:
            arxiv_id: arXiv paper ID (e.g., "1706.03762")
            pmid: PubMed ID
            doi: Digital Object Identifier

        Returns:
            Paper object or None if not found
        """
        # Try arXiv
        if arxiv_id and self.arxiv:
            try:
                paper = self.arxiv.get_paper(arxiv_id)
                if paper:
                    return self._convert_arxiv_paper(paper)
            except Exception:
                pass

        # Try PubMed
        if pmid and self.pubmed:
            try:
                paper = self.pubmed.get_paper(pmid)
                if paper:
                    return self._convert_pubmed_paper(paper)
            except Exception:
                pass

        # Try DOI via PubMed
        if doi and self.pubmed:
            try:
                paper = self.pubmed.search_by_doi(doi)
                if paper:
                    return self._convert_pubmed_paper(paper)
            except Exception:
                pass

        return None


# Convenience functions for direct use
def search_literature(query: str, max_results: int = 10) -> list[dict]:
    """
    Quick search function that returns dicts.

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        List of paper dicts
    """
    search = LiteratureSearch()
    papers = search.search(query, max_results=max_results)
    return [p.to_dict() for p in papers]


def get_paper_fulltext(
    arxiv_id: Optional[str] = None,
    pmid: Optional[str] = None,
    doi: Optional[str] = None,
) -> Optional[str]:
    """
    Get full text for a paper by identifier.

    Args:
        arxiv_id: arXiv ID
        pmid: PubMed ID
        doi: DOI

    Returns:
        Full text string or None
    """
    search = LiteratureSearch()
    paper = search.get_paper_by_id(arxiv_id=arxiv_id, pmid=pmid, doi=doi)
    if paper:
        return search.get_fulltext(paper)
    return None
