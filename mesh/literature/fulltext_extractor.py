"""
Full-text extraction module for academic papers.

Provides intelligent extraction from PDFs and HTML with:
- Full paper extraction (not just first page)
- Token-aware truncation
- Section extraction (Abstract, Methods, Results, etc.)
- Table/figure handling
- Clean text normalization
"""

import re
import subprocess
import tempfile
import os
from typing import Optional, Literal
from dataclasses import dataclass

import requests


@dataclass
class ExtractedText:
    """Result of full-text extraction."""
    text: str
    source_type: Literal["pdf", "html", "xml", "abstract"]
    char_count: int
    estimated_tokens: int
    sections: dict[str, str]  # section_name -> content
    truncated: bool = False
    original_char_count: Optional[int] = None


class FulltextExtractor:
    """
    Intelligent full-text extraction from PDFs and HTML.

    Features:
    - PDF extraction via pdftotext with layout preservation
    - Section detection and extraction
    - Token-aware truncation with smart boundaries
    - Table/figure noise reduction
    - Configurable output limits
    """

    # Default max characters (~50k tokens at ~4 chars/token)
    DEFAULT_MAX_CHARS = 200_000

    # Chars per token estimate (conservative for academic text)
    CHARS_PER_TOKEN = 4

    # Common section headers in academic papers
    SECTION_PATTERNS = [
        # Standard IMRaD sections
        (r'^(?:abstract|summary)\s*[:.]?\s*$', 'abstract'),
        (r'^(?:introduction|background)\s*[:.]?\s*$', 'introduction'),
        (r'^(?:methods?|methodology|materials?\s+and\s+methods?)\s*[:.]?\s*$', 'methods'),
        (r'^(?:results?)\s*[:.]?\s*$', 'results'),
        (r'^(?:discussion)\s*[:.]?\s*$', 'discussion'),
        (r'^(?:conclusion|conclusions|concluding\s+remarks)\s*[:.]?\s*$', 'conclusion'),
        (r'^(?:references?|bibliography)\s*[:.]?\s*$', 'references'),
        (r'^(?:acknowledgements?|acknowledgments?)\s*[:.]?\s*$', 'acknowledgements'),
        (r'^(?:appendix|appendices|supplementary)\s*[:.]?\s*$', 'appendix'),
        # Numbered sections
        (r'^[1-9]\.\s*(?:introduction|background)', 'introduction'),
        (r'^[1-9]\.\s*(?:methods?|methodology)', 'methods'),
        (r'^[1-9]\.\s*(?:results?)', 'results'),
        (r'^[1-9]\.\s*(?:discussion)', 'discussion'),
        (r'^[1-9]\.\s*(?:conclusion)', 'conclusion'),
    ]

    # Patterns for table/figure content to optionally filter
    TABLE_FIGURE_PATTERNS = [
        r'^\s*Table\s+\d+[.:]\s*',
        r'^\s*Figure\s+\d+[.:]\s*',
        r'^\s*Fig\.\s*\d+[.:]\s*',
        r'^\s*\d+\s+\d+\s+\d+\s+\d+',  # Lines of just numbers (likely tables)
    ]

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize the extractor.

        Args:
            max_chars: Maximum characters to return (default 200k)
            max_tokens: Maximum tokens (overrides max_chars if set)
        """
        if max_tokens:
            self.max_chars = max_tokens * self.CHARS_PER_TOKEN
        else:
            self.max_chars = max_chars

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
        })

    def extract_from_pdf_url(
        self,
        url: str,
        max_pages: Optional[int] = None,
        include_tables: bool = True,
        extract_sections: bool = True,
    ) -> Optional[ExtractedText]:
        """
        Download PDF and extract full text.

        Args:
            url: URL to PDF
            max_pages: Limit to first N pages (None for all)
            include_tables: Whether to keep table-like content
            extract_sections: Whether to identify sections

        Returns:
            ExtractedText or None if extraction failed
        """
        # Download PDF
        try:
            r = self.session.get(url, timeout=30, allow_redirects=True)
            r.raise_for_status()
        except Exception:
            return None

        # Check if we got a PDF
        content_type = r.headers.get("Content-Type", "").lower()
        is_pdf = "pdf" in content_type or r.content.startswith(b"%PDF-")

        if not is_pdf:
            return None

        return self.extract_from_pdf_bytes(
            r.content,
            max_pages=max_pages,
            include_tables=include_tables,
            extract_sections=extract_sections,
        )

    def extract_from_pdf_bytes(
        self,
        pdf_bytes: bytes,
        max_pages: Optional[int] = None,
        include_tables: bool = True,
        extract_sections: bool = True,
    ) -> Optional[ExtractedText]:
        """
        Extract text from PDF bytes.

        Args:
            pdf_bytes: Raw PDF content
            max_pages: Limit to first N pages (None for all)
            include_tables: Whether to keep table-like content
            extract_sections: Whether to identify sections

        Returns:
            ExtractedText or None if extraction failed
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp.flush()
                tmp_path = tmp.name

            # Build pdftotext command
            # -layout preserves document layout better
            cmd = ["pdftotext", "-layout"]
            if max_pages:
                cmd.extend(["-f", "1", "-l", str(max_pages)])
            cmd.extend([tmp_path, "-"])

            text = subprocess.check_output(cmd, text=True, timeout=60).strip()

            # Clean up temp file
            os.unlink(tmp_path)

        except Exception:
            try:
                os.unlink(tmp_path)
            except:
                pass
            return None

        if not text:
            return None

        return self._process_extracted_text(
            text,
            source_type="pdf",
            include_tables=include_tables,
            extract_sections=extract_sections,
        )

    def extract_from_html(
        self,
        html_content: str,
        extract_sections: bool = True,
    ) -> Optional[ExtractedText]:
        """
        Extract text from HTML content.

        Args:
            html_content: HTML string
            extract_sections: Whether to identify sections

        Returns:
            ExtractedText or None if extraction failed
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script, style, nav elements
            for el in soup.find_all(["script", "style", "nav", "footer", "header"]):
                el.decompose()

            # Try to find main content area
            main = (
                soup.find("main") or
                soup.find("article") or
                soup.find("div", class_=re.compile(r"content|article|paper|body", re.I)) or
                soup.body or
                soup
            )

            text = main.get_text(separator="\n", strip=True)

        except Exception:
            return None

        if not text:
            return None

        return self._process_extracted_text(
            text,
            source_type="html",
            include_tables=True,
            extract_sections=extract_sections,
        )

    def _process_extracted_text(
        self,
        text: str,
        source_type: Literal["pdf", "html", "xml", "abstract"],
        include_tables: bool = True,
        extract_sections: bool = True,
    ) -> ExtractedText:
        """
        Process and normalize extracted text.

        Args:
            text: Raw extracted text
            source_type: Where the text came from
            include_tables: Whether to keep table content
            extract_sections: Whether to identify sections

        Returns:
            Processed ExtractedText
        """
        original_char_count = len(text)

        # Normalize whitespace and clean up
        text = self._normalize_text(text)

        # Optionally filter tables/figures
        if not include_tables:
            text = self._filter_tables_figures(text)

        # Extract sections if requested
        sections = {}
        if extract_sections:
            sections = self._extract_sections(text)

        # Truncate if needed (with smart boundary detection)
        truncated = False
        if len(text) > self.max_chars:
            text = self._smart_truncate(text, self.max_chars)
            truncated = True

        char_count = len(text)
        estimated_tokens = char_count // self.CHARS_PER_TOKEN

        return ExtractedText(
            text=text,
            source_type=source_type,
            char_count=char_count,
            estimated_tokens=estimated_tokens,
            sections=sections,
            truncated=truncated,
            original_char_count=original_char_count if truncated else None,
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and clean up common PDF artifacts."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)

        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Fix hyphenation at line breaks (common PDF issue)
        text = re.sub(r'-\n(\w)', r'\1', text)

        # Remove page numbers (standalone numbers)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove form feed characters
        text = text.replace('\f', '\n\n')

        return text.strip()

    def _filter_tables_figures(self, text: str) -> str:
        """Remove table and figure content."""
        lines = text.split('\n')
        filtered = []

        for line in lines:
            skip = False
            for pattern in self.TABLE_FIGURE_PATTERNS:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            if not skip:
                filtered.append(line)

        return '\n'.join(filtered)

    def _extract_sections(self, text: str) -> dict[str, str]:
        """
        Identify and extract named sections from the text.

        Returns dict mapping section names to their content.
        """
        sections = {}
        lines = text.split('\n')

        current_section = None
        current_content = []

        for line in lines:
            line_lower = line.strip().lower()

            # Check if this line is a section header
            new_section = None
            for pattern, section_name in self.SECTION_PATTERNS:
                if re.match(pattern, line_lower):
                    new_section = section_name
                    break

            if new_section:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()

                current_section = new_section
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        Truncate text at a sensible boundary (sentence or paragraph).

        Tries to cut at:
        1. Paragraph break
        2. Sentence end (.!?)
        3. Clause break (,;:)
        4. Word boundary (space)
        """
        if len(text) <= max_chars:
            return text

        # Look for paragraph break within last 500 chars of limit
        search_start = max(0, max_chars - 500)
        search_text = text[search_start:max_chars]

        # Try paragraph break
        para_match = search_text.rfind('\n\n')
        if para_match > 0:
            return text[:search_start + para_match].strip() + "\n\n[...truncated...]"

        # Try sentence end
        sentence_match = max(
            search_text.rfind('. '),
            search_text.rfind('? '),
            search_text.rfind('! '),
        )
        if sentence_match > 0:
            return text[:search_start + sentence_match + 1].strip() + "\n\n[...truncated...]"

        # Try any newline
        newline_match = search_text.rfind('\n')
        if newline_match > 0:
            return text[:search_start + newline_match].strip() + "\n\n[...truncated...]"

        # Fall back to word boundary
        space_match = text[:max_chars].rfind(' ')
        if space_match > max_chars - 100:
            return text[:space_match].strip() + "\n\n[...truncated...]"

        # Last resort: hard cut
        return text[:max_chars].strip() + "\n\n[...truncated...]"

    def get_section(
        self,
        extracted: ExtractedText,
        section: str,
    ) -> Optional[str]:
        """
        Get a specific section from extracted text.

        Args:
            extracted: ExtractedText result
            section: Section name (e.g., "abstract", "methods", "results")

        Returns:
            Section content or None if not found
        """
        return extracted.sections.get(section.lower())

    def get_sections(
        self,
        extracted: ExtractedText,
        sections: list[str],
    ) -> str:
        """
        Get multiple sections combined.

        Args:
            extracted: ExtractedText result
            sections: List of section names to include

        Returns:
            Combined section content
        """
        parts = []
        for section in sections:
            content = extracted.sections.get(section.lower())
            if content:
                parts.append(f"## {section.title()}\n\n{content}")

        return "\n\n".join(parts)


def extract_fulltext_from_url(
    url: str,
    max_chars: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> Optional[str]:
    """
    Convenience function to extract full text from a URL.

    Args:
        url: URL to PDF or HTML
        max_chars: Maximum characters (default 200k)
        max_tokens: Maximum tokens (overrides max_chars)
        max_pages: Maximum PDF pages (None for all)

    Returns:
        Extracted text or None
    """
    extractor = FulltextExtractor(
        max_chars=max_chars or FulltextExtractor.DEFAULT_MAX_CHARS,
        max_tokens=max_tokens,
    )

    # Try PDF first
    result = extractor.extract_from_pdf_url(url, max_pages=max_pages)
    if result:
        return result.text

    # Try HTML
    try:
        r = extractor.session.get(url, timeout=30)
        r.raise_for_status()
        result = extractor.extract_from_html(r.text)
        if result:
            return result.text
    except Exception:
        pass

    return None


def extract_key_sections(
    url: str,
    sections: list[str] = ["abstract", "introduction", "methods", "results", "conclusion"],
    max_chars: Optional[int] = None,
) -> dict[str, str]:
    """
    Extract specific sections from a paper.

    Args:
        url: URL to PDF or HTML
        sections: List of section names to extract
        max_chars: Maximum characters per section

    Returns:
        Dict of section_name -> content
    """
    extractor = FulltextExtractor(max_chars=max_chars or 50000)

    result = extractor.extract_from_pdf_url(url, extract_sections=True)
    if not result:
        return {}

    return {s: result.sections.get(s, "") for s in sections if result.sections.get(s)}
