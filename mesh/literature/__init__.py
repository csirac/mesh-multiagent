# Literature search package
# Provides tools for searching academic papers across arXiv, PubMed, and Semantic Scholar

from .scholar_tool_client import ScholarToolClient
from .literature_search import LiteratureSearch, Source, Paper
from .fulltext_extractor import FulltextExtractor, ExtractedText

__all__ = [
    'ScholarToolClient',
    'LiteratureSearch',
    'Source',
    'Paper',
    'FulltextExtractor',
    'ExtractedText',
]
