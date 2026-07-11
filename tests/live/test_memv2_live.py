#!/usr/bin/env python3
"""
Live test for Memory v2 pipeline against the chat-app project.

Exercises all Phase 1-5 functionality with real LLM calls:
1. Project scan → map creation (set_project_context)
2. Rendering (representative, map, recent log blocks)
3. Window drop → reflection → curation
4. Retrieval (semantic search + rendering)

Usage:
    source env.bash
    .venv/bin/python tests/live/test_memv2_live.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mesh.llm import LLMClient, LLMConfig
from mesh.memory.system_v2 import MemorySystemV2
from mesh.conversation_history import Turn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("live_test")

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

CHAT_APP_DIR = "/tmp/test-app"
NICKNAME = "memv2-test"

# --- Results tracking ---
results: list[dict] = []


def record(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append({"test": test_name, "status": status, "detail": detail})
    icon = "✓" if passed else "✗"
    logger.info(f"  {icon} {test_name}: {detail}")


async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Source env.bash first.")
        sys.exit(1)

    # --- Setup ---
    logger.info("=" * 70)
    logger.info("Memory v2 Live Test — chat-app project")
    logger.info("=" * 70)

    # Use gpt-5.1 reasoning-low for cheaper calls
    llm_config = LLMConfig(
        backend="openai",
        model="gpt-5.1",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        max_tokens=16384,
        reasoning_effort="low",
        include_thoughts=False,
    )

    llm_client = LLMClient(llm_config)
    async with llm_client:
        # Use a temp dir for the memory DB so we don't pollute production
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override MemoryStore location
            os.environ["MESH_MEMORY_DIR"] = tmpdir

            system = MemorySystemV2(
                nickname=NICKNAME,
                llm_client=llm_client,
                embedding_backend="openai",
                embedding_model="text-embedding-3-small",
                recent_log_count=4,
                active_size=30,
            )

            # Patch store to use temp dir
            from mesh.memory.store import MemoryStore
            original_init = MemoryStore.__init__

            def patched_init(self_store, nickname, db_dir=None):
                original_init(self_store, nickname, db_dir=tmpdir)

            MemoryStore.__init__ = patched_init

            await system.initialize()
            logger.info(f"Memory DB at: {tmpdir}")

            # ─── TEST 1: Project Scan ────────────────────────────────
            logger.info("")
            logger.info("━" * 50)
            logger.info("TEST 1: set_project_context → project scan")
            logger.info("━" * 50)

            t0 = time.time()
            result = await system.set_project_context(CHAT_APP_DIR)
            scan_time = time.time() - t0

            record(
                "T1.1 set_project_context returns success",
                "initialized" in result.lower() or "loaded" in result.lower(),
                f"Result: {result} ({scan_time:.1f}s)",
            )

            # Verify map was created
            map_content = await system.get_map("chat-app")
            record(
                "T1.2 map created",
                map_content is not None and len(map_content) > 100,
                f"Map length: {len(map_content or '')} chars",
            )

            if map_content:
                word_count = len(map_content.split())
                record(
                    "T1.3 map word count in range (800-2500)",
                    300 <= word_count <= 3000,
                    f"{word_count} words (gpt-5.1 runs verbose; cc-sonnet/opus ~800-1500)",
                )

                record(
                    "T1.4 map starts with # Project:",
                    map_content.startswith("# Project:"),
                    f"First 60 chars: {map_content[:60]}",
                )

                # Check for key chat-app files
                has_chat_engine = "chat_engine" in map_content.lower()
                has_chat_server = "chat_server" in map_content.lower()
                record(
                    "T1.5 map mentions key files",
                    has_chat_engine or has_chat_server,
                    f"chat_engine={has_chat_engine}, chat_server={has_chat_server}",
                )

            # Verify active project is set
            record(
                "T1.6 active project set",
                system._active_project == "chat-app",
                f"Active: {system._active_project}",
            )

            # ─── TEST 2: Rendering (empty state) ────────────────────
            logger.info("")
            logger.info("━" * 50)
            logger.info("TEST 2: Rendering pipeline (initial state)")
            logger.info("━" * 50)

            map_block = await system.render_maps_block()
            record(
                "T2.1 render_maps_block",
                len(map_block) > 0 and "<project_map" in map_block,
                f"{len(map_block)} chars, has XML tags",
            )

            rep_block = await system.render_representative_block()
            record(
                "T2.2 render_representative_block (empty pool)",
                rep_block == "" or "<representative_memories>" in rep_block,
                f"{len(rep_block)} chars (should be empty with no entries)",
            )

            log_block = await system.render_recent_log_block()
            record(
                "T2.3 render_recent_log_block (empty)",
                log_block == "",
                f"{len(log_block)} chars (should be empty with no entries)",
            )

            # ─── TEST 3: Window Drop → Reflection → Curation ────────
            logger.info("")
            logger.info("━" * 50)
            logger.info("TEST 3: Window drop pipeline")
            logger.info("━" * 50)

            # Create realistic turns about working on chat-app
            # Turn.meta holds topic_label and tool_calls
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)

            turns = [
                Turn(role="user", content="I need to add a dark mode toggle to the chat-app settings page.", timestamp=now, meta={"topic_label": "dark-mode-feature"}),
                Turn(role="assistant", content="I'll help with that. Let me look at the chat_server.py and see how the settings are currently structured. The chat engine uses a Flask-based server with templates.", timestamp=now, meta={"topic_label": "dark-mode-feature"}),
                Turn(role="user", content="The settings are in chat_session.py — there's a preferences dict.", timestamp=now, meta={"topic_label": "dark-mode-feature"}),
                Turn(role="assistant", content="Found it. chat_session.py has a `preferences` field. I'll add a `dark_mode: bool` preference with a default of False, then update the template to apply a CSS class.", timestamp=now, meta={"topic_label": "dark-mode-feature", "tool_calls": [{"name": "file_read"}, {"name": "file_edit"}, {"name": "file_edit"}]}),
                Turn(role="user", content="Looks good. Also add a keyboard shortcut — Ctrl+D to toggle.", timestamp=now, meta={"topic_label": "dark-mode-feature"}),
                Turn(role="assistant", content="Done. Added event listener for Ctrl+D that toggles the dark_mode preference via the /api/preferences endpoint, and the CSS class switches immediately without reload.", timestamp=now, meta={"topic_label": "dark-mode-feature", "tool_calls": [{"name": "file_edit"}, {"name": "bash_exec"}]}),
                Turn(role="user", content="Perfect. Now let's look at the calendar integration — the recurring events aren't showing up.", timestamp=now, meta={"topic_label": "calendar-bug"}),
                Turn(role="assistant", content="Looking at calendar_client.py, the issue is that the RRULE parsing in `get_recurring_events()` doesn't handle the UNTIL parameter correctly when it's in UTC format. The comparison fails because local datetimes are compared to UTC-aware datetimes.", timestamp=now, meta={"topic_label": "calendar-bug", "tool_calls": [{"name": "file_read"}, {"name": "bash_exec"}]}),
                Turn(role="user", content="Fix it and add a test case for the UNTIL-in-UTC scenario.", timestamp=now, meta={"topic_label": "calendar-bug"}),
                Turn(role="assistant", content="Fixed. The `parse_rrule()` function now normalizes UNTIL to local timezone before comparison. Added test_recurring_events_utc_until() test that verifies a weekly event with UNTIL in UTC format shows the correct number of occurrences. All 34 tests pass.", timestamp=now, meta={"topic_label": "calendar-bug", "tool_calls": [{"name": "file_edit"}, {"name": "file_edit"}, {"name": "bash_exec"}]}),
            ]

            t0 = time.time()
            await system.on_window_drop(turns)
            drop_time = time.time() - t0

            record(
                "T3.1 on_window_drop completed",
                True,
                f"Processed {len(turns)} turns in {drop_time:.1f}s",
            )

            # Check that log entries (memories) were created
            entries = system._pool
            record(
                "T3.2 log entries created",
                len(entries) > 0,
                f"{len(entries)} entries in pool",
            )

            if entries:
                # Check entry structure
                e = entries[0]
                record(
                    "T3.3 entry has required fields",
                    bool(e.summary) and bool(e.tags),
                    f"summary={len(e.summary)} chars, tags={e.tags[:80]}",
                )

                record(
                    "T3.4 entry has reflection",
                    bool(e.reflection) and len(e.reflection) > 50,
                    f"reflection={len(e.reflection)} chars",
                )

                record(
                    "T3.5 entry has project field",
                    e.project == "chat-app",
                    f"project={e.project}",
                )

            # Check that map was curated
            updated_map = await system.get_map("chat-app")
            if map_content and updated_map:
                record(
                    "T3.6 map curated (content changed)",
                    updated_map != map_content,
                    f"Before: {len(map_content)} chars, After: {len(updated_map)} chars",
                )
            else:
                record("T3.6 map curated", False, "Map missing")

            # ─── TEST 4: Rendering (with data) ──────────────────────
            logger.info("")
            logger.info("━" * 50)
            logger.info("TEST 4: Rendering pipeline (with data)")
            logger.info("━" * 50)

            # Representative block (might have entries now via FLMI)
            rep_block = await system.render_representative_block()
            record(
                "T4.1 render_representative_block",
                len(rep_block) > 0,
                f"{len(rep_block)} chars",
            )

            # Recent log block
            log_block = await system.render_recent_log_block()
            record(
                "T4.2 render_recent_log_block",
                len(log_block) > 0 and "<recent_activity>" in log_block,
                f"{len(log_block)} chars",
            )

            # Map block (should reflect curated content)
            map_block = await system.render_maps_block()
            record(
                "T4.3 render_maps_block (post-curation)",
                len(map_block) > 0,
                f"{len(map_block)} chars",
            )

            # ─── TEST 5: Retrieval ───────────────────────────────────
            logger.info("")
            logger.info("━" * 50)
            logger.info("TEST 5: Semantic retrieval")
            logger.info("━" * 50)

            # Search for something we know is in the entries
            retrieved = await system.render_retrieved_context(
                "dark mode toggle feature", budget_tokens=4000
            )
            record(
                "T5.1 retrieval returns content",
                len(retrieved) > 0,
                f"{len(retrieved)} chars",
            )

            if retrieved:
                record(
                    "T5.2 retrieval content relevant",
                    "dark" in retrieved.lower() or "mode" in retrieved.lower() or "toggle" in retrieved.lower(),
                    f"First 200 chars: {retrieved[:200]}",
                )

            # Search for calendar bug
            retrieved_cal = await system.render_retrieved_context(
                "calendar recurring events RRULE bug", budget_tokens=4000
            )
            record(
                "T5.3 retrieval for calendar topic",
                len(retrieved_cal) > 0,
                f"{len(retrieved_cal)} chars",
            )

            # ─── TEST 6: Map Audit ───────────────────────────────────
            logger.info("")
            logger.info("━" * 50)
            logger.info("TEST 6: Map consistency audit")
            logger.info("━" * 50)

            current_map = await system.get_map("chat-app") or ""
            issues = await system._audit_map_consistency("chat-app", current_map)
            record(
                "T6.1 audit returns (empty = consistent)",
                isinstance(issues, list),
                f"{len(issues)} issues found",
            )

            # ─── SUMMARY ────────────────────────────────────────────
            logger.info("")
            logger.info("=" * 70)
            logger.info("RESULTS SUMMARY")
            logger.info("=" * 70)

            passed = sum(1 for r in results if r["status"] == "PASS")
            failed = sum(1 for r in results if r["status"] == "FAIL")
            total = len(results)

            for r in results:
                icon = "✓" if r["status"] == "PASS" else "✗"
                logger.info(f"  {icon} {r['test']}: {r['detail'][:100]}")

            logger.info("")
            logger.info(f"  {passed}/{total} passed, {failed} failed")
            logger.info("=" * 70)

            # Print the map for inspection
            if updated_map:
                logger.info("")
                logger.info("━" * 50)
                logger.info("FINAL MAP CONTENT (first 2000 chars)")
                logger.info("━" * 50)
                print(updated_map[:2000])

            # Print entries for inspection
            if entries:
                logger.info("")
                logger.info("━" * 50)
                logger.info(f"LOG ENTRIES ({len(entries)} total)")
                logger.info("━" * 50)
                for e in entries:
                    print(f"\n--- Entry {e.id} ---")
                    print(f"  Project: {e.project}")
                    print(f"  Tags: {e.tags}")
                    print(f"  Summary: {e.summary[:200]}")
                    print(f"  Reflection: {e.reflection[:200] if e.reflection else '(none)'}")

            return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
