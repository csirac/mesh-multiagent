"""
Mesh clients package.

Contains self-contained client implementations for external services
(Gmail, Calendar, Exa, Browser, Bash).
"""

from .gmail_client import GmailClient
from .calendar_client import CalendarClient
from .account_manager import AccountManager, AccountConfig, ToolHost
from .exa_client import ExaSearchClient
from .bash_tools import BashTools

try:
    from .browser_client_minimal import BrowserClient
except ImportError:
    BrowserClient = None  # type: ignore[assignment,misc]

__all__ = [
    "GmailClient",
    "CalendarClient",
    "AccountManager",
    "AccountConfig",
    "ToolHost",
    "ExaSearchClient",
    "BashTools",
    "BrowserClient",
]
