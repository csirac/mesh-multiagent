"""
Shared CC usage utilities for the mesh.

Provides async and sync functions to fetch Claude Code usage data
from Anthropic's OAuth usage API. Used by:
- Router's CCUsageMonitor (async)
- cc-usage CLI script (sync)
- mesh/llm.py fallback (sync)
- tool_implementations (sync wrapper)
"""

from __future__ import annotations

import json
import logging
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Anthropic requires this header to distinguish CC clients from generic API callers.
# Without it, the usage endpoint returns 429.
_CC_USER_AGENT = "claude-code/2.1.69"


@dataclass
class CCUsageResult:
    """Result of fetching usage for a single account."""

    label: str
    home_dir: str
    sub_type: str = "unknown"
    email: str = ""
    windows: dict[str, float] = field(default_factory=dict)  # window_name -> utilization %
    resets: dict[str, str] = field(default_factory=dict)  # window_name -> reset time delta
    error: str | None = None


def _format_reset_delta(resets_at: str) -> str:
    """Format a reset time as a human-readable delta."""
    if not resets_at:
        return ""
    try:
        reset_dt = datetime.fromisoformat(resets_at.replace("Z", "+00:00"))
        delta = reset_dt - datetime.now(timezone.utc)
        total_secs = int(delta.total_seconds())
        if total_secs <= 0:
            return "now"
        days = total_secs // 86400
        hours = (total_secs % 86400) // 3600
        mins = (total_secs % 3600) // 60
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if mins:
            parts.append(f"{mins}m")
        return " ".join(parts) or "<1m"
    except Exception:
        return resets_at


def _derive_creds_path(home_dir: Path) -> Path:
    """Derive credentials path from HOME dir (home/.claude/.credentials.json)."""
    return home_dir / ".claude" / ".credentials.json"


def _read_account_email(home_dir: Path) -> str:
    """Read email address from HOME/.claude.json oauthAccount section."""
    claude_json = home_dir / ".claude.json"
    try:
        data = json.loads(claude_json.read_text())
        return data.get("oauthAccount", {}).get("emailAddress", "")
    except Exception:
        return ""


def _refresh_token_sync(oauth: dict, creds: dict, creds_path: Path) -> str:
    """Synchronously refresh token if expired. Returns access token."""
    import httpx

    access_token = oauth.get("accessToken", "")
    refresh_token = oauth.get("refreshToken", "")
    expires_at_ms = oauth.get("expiresAt", 0)

    now_ms = int(time.time() * 1000)
    if expires_at_ms > 0 and now_ms > expires_at_ms - 600_000 and refresh_token:
        try:
            r = httpx.post(
                "https://api.anthropic.com/v1/oauth/token",
                json={"grant_type": "refresh_token", "refresh_token": refresh_token},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if r.status_code == 200:
                token_data = r.json()
                access_token = token_data.get("access_token", access_token)
                new_refresh = token_data.get("refresh_token", refresh_token)
                new_expires_in = token_data.get("expires_in", 3600)
                new_expires_at = int(time.time() * 1000) + (new_expires_in * 1000)
                # Persist refreshed token
                try:
                    oauth["accessToken"] = access_token
                    oauth["refreshToken"] = new_refresh
                    oauth["expiresAt"] = new_expires_at
                    creds["claudeAiOauth"] = oauth
                    tmp_path = creds_path.with_suffix(".tmp")
                    tmp_path.write_text(json.dumps(creds, indent=2))
                    tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                    tmp_path.rename(creds_path)
                except Exception as e:
                    logger.warning(f"Failed to persist refreshed token: {e}")
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")

    return access_token


async def _refresh_token_async(oauth: dict, creds: dict, creds_path: Path) -> str:
    """Asynchronously refresh token if expired. Returns access token."""
    import httpx

    access_token = oauth.get("accessToken", "")
    refresh_token = oauth.get("refreshToken", "")
    expires_at_ms = oauth.get("expiresAt", 0)

    now_ms = int(time.time() * 1000)
    if expires_at_ms > 0 and now_ms > expires_at_ms - 600_000 and refresh_token:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/oauth/token",
                    json={"grant_type": "refresh_token", "refresh_token": refresh_token},
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
            if r.status_code == 200:
                token_data = r.json()
                access_token = token_data.get("access_token", access_token)
                new_refresh = token_data.get("refresh_token", refresh_token)
                new_expires_in = token_data.get("expires_in", 3600)
                new_expires_at = int(time.time() * 1000) + (new_expires_in * 1000)
                # Persist refreshed token
                try:
                    oauth["accessToken"] = access_token
                    oauth["refreshToken"] = new_refresh
                    oauth["expiresAt"] = new_expires_at
                    creds["claudeAiOauth"] = oauth
                    tmp_path = creds_path.with_suffix(".tmp")
                    tmp_path.write_text(json.dumps(creds, indent=2))
                    tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                    tmp_path.rename(creds_path)
                except Exception as e:
                    logger.warning(f"Failed to persist refreshed token: {e}")
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")

    return access_token


def fetch_account_usage_sync(home_dir: str, label: str = "") -> CCUsageResult:
    """
    Fetch CC usage for a single account synchronously.

    Args:
        home_dir: HOME directory path (e.g., "~/.claude-acct2" or "~")
        label: Human-readable label for the account

    Returns:
        CCUsageResult with utilization data or error
    """
    import httpx

    home = Path(home_dir).expanduser()
    creds_path = _derive_creds_path(home)

    result = CCUsageResult(
        label=label or home_dir,
        home_dir=str(home),
    )

    if not creds_path.exists():
        result.error = "no credentials file"
        return result

    try:
        creds = json.loads(creds_path.read_text())
    except Exception as e:
        result.error = f"bad credentials: {e}"
        return result

    oauth = creds.get("claudeAiOauth")
    if not oauth or not isinstance(oauth, dict):
        result.error = "no OAuth section"
        return result

    result.sub_type = oauth.get("subscriptionType", "unknown")
    result.email = _read_account_email(home)
    access_token = _refresh_token_sync(oauth, creds, creds_path)

    if not access_token:
        result.error = "no access token"
        return result

    data = None
    for attempt in range(3):
        try:
            r = httpx.get(
                "https://api.anthropic.com/api/oauth/usage",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "anthropic-beta": "oauth-2025-04-20",
                    "Content-Type": "application/json",
                    "User-Agent": _CC_USER_AGENT,
                },
                timeout=15,
            )
            if r.status_code == 401:
                result.error = "token expired (401)"
                return result
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 30 * (attempt + 1)))
                if attempt < 2:
                    time.sleep(min(retry_after, 60))
                    continue
                result.error = "rate limited (429)"
                return result
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 2:
                result.error = str(e)
                return result
            time.sleep(5 * (attempt + 1))

    if data is None:
        result.error = "no data after retries"
        return result

    # Extract window utilization
    window_keys = ["five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"]
    for key in window_keys:
        info = data.get(key)
        if info and info.get("utilization") is not None:
            result.windows[key] = info["utilization"]
            if info.get("resets_at"):
                result.resets[key] = _format_reset_delta(info["resets_at"])

    return result


async def fetch_account_usage_async(home_dir: str, label: str = "") -> CCUsageResult:
    """
    Fetch CC usage for a single account asynchronously.

    Args:
        home_dir: HOME directory path (e.g., "~/.claude-acct2" or "~")
        label: Human-readable label for the account

    Returns:
        CCUsageResult with utilization data or error
    """
    import httpx

    home = Path(home_dir).expanduser()
    creds_path = _derive_creds_path(home)

    result = CCUsageResult(
        label=label or home_dir,
        home_dir=str(home),
    )

    if not creds_path.exists():
        result.error = "no credentials file"
        return result

    try:
        creds = json.loads(creds_path.read_text())
    except Exception as e:
        result.error = f"bad credentials: {e}"
        return result

    oauth = creds.get("claudeAiOauth")
    if not oauth or not isinstance(oauth, dict):
        result.error = "no OAuth section"
        return result

    result.sub_type = oauth.get("subscriptionType", "unknown")
    result.email = _read_account_email(home)
    access_token = await _refresh_token_async(oauth, creds, creds_path)

    if not access_token:
        result.error = "no access token"
        return result

    import asyncio as _asyncio

    data = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.anthropic.com/api/oauth/usage",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "anthropic-beta": "oauth-2025-04-20",
                        "Content-Type": "application/json",
                        "User-Agent": _CC_USER_AGENT,
                    },
                    timeout=15,
                )
            if r.status_code == 401:
                result.error = "token expired (401)"
                return result
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 30 * (attempt + 1)))
                if attempt < 2:
                    await _asyncio.sleep(min(retry_after, 60))
                    continue
                result.error = "rate limited (429)"
                return result
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 2:
                result.error = str(e)
                return result
            await _asyncio.sleep(5 * (attempt + 1))

    if data is None:
        result.error = "no data after retries"
        return result

    # Extract window utilization
    window_keys = ["five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"]
    for key in window_keys:
        info = data.get(key)
        if info and info.get("utilization") is not None:
            result.windows[key] = info["utilization"]
            if info.get("resets_at"):
                result.resets[key] = _format_reset_delta(info["resets_at"])

    return result


async def fetch_all_usage_async(
    home_dirs: list[tuple[str, str]],  # (label, home_dir)
) -> list[CCUsageResult]:
    """
    Fetch usage for multiple accounts sequentially with a small delay
    between each to avoid rate limiting.

    Args:
        home_dirs: List of (label, home_dir) tuples

    Returns:
        List of CCUsageResult in same order as input
    """
    import asyncio

    results: list[CCUsageResult] = []
    for i, (label, home_dir) in enumerate(home_dirs):
        if i > 0:
            await asyncio.sleep(5)  # Stagger requests to avoid 429s
        results.append(await fetch_account_usage_async(home_dir, label))
    return results


def format_usage_compact(result: CCUsageResult) -> str:
    """Format a single account's usage as a compact string."""
    acct_label = result.email or result.label
    if result.error:
        return f"{acct_label}[{result.sub_type}]: {result.error}"

    label_map = {
        "five_hour": "5h",
        "seven_day": "7d",
        "seven_day_opus": "opus",
        "seven_day_sonnet": "sonnet",
    }

    parts = []
    for key, label in label_map.items():
        if key in result.windows:
            parts.append(f"{label}: {result.windows[key]:.0f}%")

    usage_str = " | ".join(parts) if parts else "no data"
    return f"{acct_label}[{result.sub_type}] {usage_str}"


def format_usage_summary(results: list[CCUsageResult]) -> str:
    """Format multiple accounts as a single summary line."""
    parts = [format_usage_compact(r) for r in results]
    return "   |   ".join(parts)


def format_colorized(result: CCUsageResult, width: int = 20) -> str:
    """Format usage with progress bar and color based on utilization."""
    # ANSI color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    acct_header = result.label
    if result.email:
        acct_header += f" ({result.email})"

    if result.error:
        return f"{acct_header} [{result.sub_type}]\n  {RED}Error: {result.error}{RESET}"

    lines = [f"{BOLD}{acct_header} [{result.sub_type}]{RESET}"]

    label_map = {
        "five_hour": ("5-hour", "5h"),
        "seven_day": ("7-day", "7d"),
        "seven_day_opus": ("7d-opus", "opus"),
        "seven_day_sonnet": ("7d-sonnet", "sonnet"),
    }

    for key, (name, short) in label_map.items():
        if key not in result.windows:
            continue
        pct = result.windows[key]
        filled = int(pct / 100 * width)
        bar = "[" + "#" * filled + "." * (width - filled) + "]"
        reset = result.resets.get(key, "")

        line = f"  {name:<12} {bar} {pct:5.1f}%"
        if reset:
            line += f"   resets: {reset}"

        # Colorize based on utilization
        if pct >= 90:
            line = f"{RED}{line}{RESET}"
        elif pct >= 70:
            line = f"{YELLOW}{line}{RESET}"
        else:
            line = f"{GREEN}{line}{RESET}"

        lines.append(line)

    return "\n".join(lines)
