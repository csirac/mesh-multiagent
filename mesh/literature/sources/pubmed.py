"""
PubMed and PubMed Central (PMC) API client.

Provides search and full-text retrieval for biomedical literature.

APIs used:
- E-utilities (ESearch, EFetch, ESummary) for PubMed search and metadata
- BioC API for PMC full-text retrieval
- PMC OA Service for checking Open Access availability

References:
- https://www.ncbi.nlm.nih.gov/books/NBK25497/
- https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/
- https://pmc.ncbi.nlm.nih.gov/tools/oa-service/
"""

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus

import requests


@dataclass
class PubMedPaper:
    """Represents a paper from PubMed/PMC."""
    pmid: str
    title: str
    authors: list[str]
    abstract: str
    journal: str
    pub_date: str
    doi: Optional[str] = None
    pmcid: Optional[str] = None  # PMC ID if available (for full text)
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
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "doi": self.doi,
            "pmcid": self.pmcid,
            "keywords": self.keywords,
            "mesh_terms": self.mesh_terms,
            "pubmed_url": self.pubmed_url,
            "pmc_url": self.pmc_url,
        }


class PubMedClient:
    """
    Client for PubMed and PMC APIs.

    Features:
    - Search PubMed with complex queries
    - Fetch paper metadata and abstracts
    - Retrieve full text from PMC (when available)
    - Rate limiting and API key support
    """

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    BIOC_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
    PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        rate_limit: float = 0.34,  # ~3 requests/sec without API key
    ):
        """
        Initialize the PubMed client.

        Args:
            api_key: NCBI API key (optional, allows 10 req/s instead of 3)
            email: Email for NCBI identification (recommended)
            rate_limit: Minimum seconds between requests
        """
        self.api_key = api_key
        self.email = email
        self.rate_limit = rate_limit if not api_key else 0.1  # 10 req/s with key
        self._last_request_time = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PubMedClient/1.0 (Python; research assistant)"
        })

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _add_auth_params(self, params: dict) -> dict:
        """Add API key and email to request parameters."""
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> list[PubMedPaper]:
        """
        Search PubMed for papers matching a query.

        Args:
            query: Search query (supports PubMed syntax: field tags, boolean ops)
            max_results: Maximum number of results to return
            sort: Sort order - "relevance", "pub_date", "first_author"
            min_date: Minimum publication date (YYYY/MM/DD or YYYY)
            max_date: Maximum publication date (YYYY/MM/DD or YYYY)

        Returns:
            List of PubMedPaper objects with metadata and abstracts

        Example queries:
            - "machine learning cancer diagnosis"
            - "COVID-19[Title] AND vaccine[Title]"
            - "Smith J[Author] AND 2023[PDAT]"
        """
        # Step 1: ESearch to get PMIDs
        self._rate_limit()

        search_params = self._add_auth_params({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": sort,
        })

        if min_date:
            search_params["mindate"] = min_date
        if max_date:
            search_params["maxdate"] = max_date
        if min_date or max_date:
            search_params["datetype"] = "pdat"  # publication date

        resp = self.session.get(
            f"{self.EUTILS_BASE}/esearch.fcgi",
            params=search_params,
            timeout=30,
        )
        resp.raise_for_status()

        search_result = resp.json()
        id_list = search_result.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return []

        # Step 2: EFetch to get full records
        return self._fetch_papers(id_list)

    def _fetch_papers(self, pmids: list[str]) -> list[PubMedPaper]:
        """Fetch full paper details for a list of PMIDs."""
        self._rate_limit()

        fetch_params = self._add_auth_params({
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        })

        resp = self.session.get(
            f"{self.EUTILS_BASE}/efetch.fcgi",
            params=fetch_params,
            timeout=60,
        )
        resp.raise_for_status()

        return self._parse_pubmed_xml(resp.text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[PubMedPaper]:
        """Parse PubMed XML response into PubMedPaper objects."""
        papers = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers

        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception:
                continue

        return papers

    def _parse_article(self, article: ET.Element) -> Optional[PubMedPaper]:
        """Parse a single PubmedArticle element."""
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None

        # PMID
        pmid_elem = medline.find(".//PMID")
        if pmid_elem is None or pmid_elem.text is None:
            return None
        pmid = pmid_elem.text

        # Article info
        art = medline.find(".//Article")
        if art is None:
            return None

        # Title
        title_elem = art.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        # Abstract
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

        # Authors
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

        # Journal
        journal_elem = art.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None and journal_elem.text else ""

        # Publication date
        pub_date_parts = []
        pub_date_elem = art.find(".//Journal/JournalIssue/PubDate")
        if pub_date_elem is not None:
            year = pub_date_elem.find("Year")
            month = pub_date_elem.find("Month")
            day = pub_date_elem.find("Day")
            if year is not None and year.text:
                pub_date_parts.append(year.text)
            if month is not None and month.text:
                pub_date_parts.append(month.text)
            if day is not None and day.text:
                pub_date_parts.append(day.text)
        pub_date = " ".join(pub_date_parts)

        # DOI
        doi = None
        for eloc in art.findall(".//ELocationID"):
            if eloc.get("EIdType") == "doi":
                doi = eloc.text
                break

        # PMC ID (from ArticleIdList in PubmedData)
        pmcid = None
        pubmed_data = article.find(".//PubmedData")
        if pubmed_data is not None:
            for art_id in pubmed_data.findall(".//ArticleId"):
                if art_id.get("IdType") == "pmc":
                    pmcid = art_id.text
                    break

        # Keywords
        keywords = []
        keyword_list = medline.find(".//KeywordList")
        if keyword_list is not None:
            for kw in keyword_list.findall(".//Keyword"):
                if kw.text:
                    keywords.append(kw.text)

        # MeSH terms
        mesh_terms = []
        mesh_list = medline.find(".//MeshHeadingList")
        if mesh_list is not None:
            for mesh in mesh_list.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)

        return PubMedPaper(
            pmid=pmid,
            title=title,
            authors=authors,
            abstract=abstract,
            journal=journal,
            pub_date=pub_date,
            doi=doi,
            pmcid=pmcid,
            keywords=keywords,
            mesh_terms=mesh_terms,
        )

    def get_paper(self, pmid: str) -> Optional[PubMedPaper]:
        """
        Fetch a single paper by PMID.

        Args:
            pmid: PubMed ID (numeric string)

        Returns:
            PubMedPaper object or None if not found
        """
        papers = self._fetch_papers([pmid])
        return papers[0] if papers else None

    def get_fulltext(
        self,
        pmid: Optional[str] = None,
        pmcid: Optional[str] = None,
        format: str = "json",
    ) -> Optional[str]:
        """
        Retrieve full text from PMC using BioC API.

        Only works for papers in the PMC Open Access Subset.

        Args:
            pmid: PubMed ID
            pmcid: PMC ID (e.g., "PMC1234567")
            format: "json" or "xml"

        Returns:
            Full text as string, or None if not available
        """
        if not pmid and not pmcid:
            raise ValueError("Must provide either pmid or pmcid")

        # Use pmcid if provided, otherwise use pmid
        identifier = pmcid if pmcid else pmid

        self._rate_limit()

        url = f"{self.BIOC_BASE}/BioC_{format}/{identifier}/unicode"

        try:
            resp = self.session.get(url, timeout=60)

            if resp.status_code == 404:
                return None

            resp.raise_for_status()

            if format == "json":
                return self._extract_text_from_bioc_json(resp.text)
            else:
                return self._extract_text_from_bioc_xml(resp.text)

        except requests.RequestException:
            return None

    def _extract_text_from_bioc_json(self, json_text: str) -> str:
        """Extract readable text from BioC JSON format."""
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return json_text

        text_parts = []

        # BioC JSON structure: list of collections, each with documents
        # Structure: list[0]["documents"][0]["passages"]
        collections = data if isinstance(data, list) else [data]

        for collection in collections:
            documents = collection.get("documents", [])
            for doc in documents:
                passages = doc.get("passages", [])
                for passage in passages:
                    # Get section type
                    infons = passage.get("infons", {})
                    section_type = infons.get("section_type", "")

                    text = passage.get("text", "")
                    if text:
                        if section_type:
                            text_parts.append(f"\n[{section_type}]\n{text}")
                        else:
                            text_parts.append(text)

        return "\n".join(text_parts)

    def _extract_text_from_bioc_xml(self, xml_text: str) -> str:
        """Extract readable text from BioC XML format."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return xml_text

        text_parts = []

        for passage in root.findall(".//passage"):
            section_type = ""
            for infon in passage.findall("infon"):
                if infon.get("key") == "section_type":
                    section_type = infon.text or ""
                    break

            text_elem = passage.find("text")
            if text_elem is not None and text_elem.text:
                if section_type:
                    text_parts.append(f"\n[{section_type}]\n{text_elem.text}")
                else:
                    text_parts.append(text_elem.text)

        return "\n".join(text_parts)

    def check_oa_availability(self, pmcid: str) -> dict:
        """
        Check if a PMC article is available in the Open Access subset.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567")

        Returns:
            Dict with availability info and download links
        """
        self._rate_limit()

        params = {"id": pmcid}

        try:
            resp = self.session.get(
                self.PMC_OA_BASE,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()

            return self._parse_oa_response(resp.text)

        except requests.RequestException:
            return {"available": False, "error": "Request failed"}

    def _parse_oa_response(self, xml_text: str) -> dict:
        """Parse PMC OA service response."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return {"available": False, "error": "Parse error"}

        record = root.find(".//record")
        if record is None:
            # Check for error
            error = root.find(".//error")
            if error is not None:
                return {"available": False, "error": error.text}
            return {"available": False}

        result = {"available": True}

        # Get citation info
        citation = record.get("citation", "")
        if citation:
            result["citation"] = citation

        # Get download links
        links = []
        for link in record.findall(".//link"):
            link_info = {
                "format": link.get("format", ""),
                "href": link.get("href", ""),
            }
            if link_info["href"]:
                links.append(link_info)

        if links:
            result["links"] = links

        return result

    def search_by_author(
        self,
        author: str,
        max_results: int = 10,
    ) -> list[PubMedPaper]:
        """
        Search for papers by author name.

        Args:
            author: Author name (e.g., "Smith J" or "John Smith")
            max_results: Maximum number of results

        Returns:
            List of PubMedPaper objects
        """
        query = f"{author}[Author]"
        return self.search(query, max_results=max_results)

    def search_by_doi(self, doi: str) -> Optional[PubMedPaper]:
        """
        Find a paper by DOI.

        Args:
            doi: Digital Object Identifier

        Returns:
            PubMedPaper object or None
        """
        query = f"{doi}[DOI]"
        papers = self.search(query, max_results=1)
        return papers[0] if papers else None

    def get_related_papers(
        self,
        pmid: str,
        max_results: int = 10,
    ) -> list[PubMedPaper]:
        """
        Get papers related to a given PMID using ELink.

        Args:
            pmid: PubMed ID of source paper
            max_results: Maximum number of related papers

        Returns:
            List of related PubMedPaper objects
        """
        self._rate_limit()

        # Use ELink to find related articles
        link_params = self._add_auth_params({
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "cmd": "neighbor_score",
            "retmode": "json",
        })

        # Retry up to 3 times for transient server errors
        for attempt in range(3):
            try:
                resp = self.session.get(
                    f"{self.EUTILS_BASE}/elink.fcgi",
                    params=link_params,
                    timeout=30,
                )
                resp.raise_for_status()

                # Parse JSON, handling potential encoding issues
                try:
                    link_data = resp.json()
                except Exception:
                    # Try with strict=False for control characters
                    link_data = json.loads(resp.text, strict=False)

                # Check for API error
                if "ERROR" in link_data:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    return []

                # Extract related PMIDs
                related_pmids = []
                linksets = link_data.get("linksets", [])
                if linksets:
                    linksetdbs = linksets[0].get("linksetdbs", [])
                    for lsdb in linksetdbs:
                        if lsdb.get("linkname") == "pubmed_pubmed":
                            links = lsdb.get("links", [])
                            related_pmids = [str(link) for link in links[:max_results]]
                            break

                if not related_pmids:
                    return []

                return self._fetch_papers(related_pmids)

            except requests.RequestException:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return []

        return []


# Convenience functions for direct use
def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """Search PubMed and return results as dictionaries."""
    client = PubMedClient()
    papers = client.search(query, max_results=max_results)
    return [p.to_dict() for p in papers]


def get_pubmed_fulltext(pmid: str = None, pmcid: str = None) -> Optional[str]:
    """Get full text from PMC if available."""
    client = PubMedClient()
    return client.get_fulltext(pmid=pmid, pmcid=pmcid)
