#!/usr/bin/env python3
"""
Live integration test: CC session context survives account rotation.

Creates a CCSession with 2 accounts, sends a turn on account 1 establishing
a unique fact, forces rotation to account 2, sends a turn asking about
the fact, and verifies the response references it.

Usage:
    python tests/test_cc_session_rotation.py

Requires: claude CLI authenticated on at least 2 accounts.
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh.cc_session import CCSession, CCStreamEvent
from mesh.config import CCSessionConfig, MeshConfig, load_config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("test_rotation")


async def collect_text(session: CCSession, prompt: str) -> str:
    """Send a turn and collect all text output."""
    text_parts = []
    async for event in session.send_turn(prompt):
        if event.type == "text" and event.content:
            text_parts.append(event.content)
        elif event.type == "session_id":
            logger.info(f"Session ID: {event.session_id}")
        elif event.type == "error":
            logger.error(f"Error: {event.content}")
    return "\n".join(text_parts)


async def run_test():
    """Run the rotation test."""
    # Load config to get fallback homes
    config = load_config()
    if not config:
        logger.error("No mesh.yaml found")
        return False

    backend = config.llm_backends.get("claude-code")
    if not backend:
        logger.error("No claude-code backend in config")
        return False

    from mesh.config import backend_config_to_llm_config
    llm_config = backend_config_to_llm_config(backend)

    # Use only 2 accounts for the test
    # Default (None) + first fallback
    original_homes = llm_config.cc_fallback_homes
    if not original_homes:
        logger.error("No cc_fallback_homes configured")
        return False

    # Use default (None) + acct3 — both verified not rate-limited.
    # The _build_env fix ensures default uses _real_user_home ($HOME)
    # even when this test runs inside a CC session with overridden $HOME.
    test_fallback = [h for h in original_homes if "acct3" in h]
    if not test_fallback:
        test_fallback = [original_homes[1]]  # fallback
    llm_config.cc_fallback_homes = test_fallback
    logger.info(f"Testing with accounts: default + {test_fallback[0]}")

    cc_config = CCSessionConfig(
        cc_model="haiku",  # Use haiku for speed/cheapness
        cc_max_turns=2,    # Just enough for one response
        session_dir="/tmp/mesh-test/sessions",
    )

    nickname = "test-rotation"
    session = CCSession(
        nickname=nickname,
        agent_type="sysadmin",
        node_id="agent:sysadmin:test-rotation",
        config=cc_config,
        llm_config=llm_config,
        memory_system=None,
        identity_block="You are a test agent. Keep responses short (1-2 sentences).",
        personality_block="",
        mesh_protocol_block="",
    )

    # No _all_homes override needed — CCSession.__init__ builds
    # [None, expanded_acct3] from llm_config.cc_fallback_homes

    # Clean up any prior session state for this test
    sessions_file = Path("/tmp/mesh-test/sessions") / f"{nickname}.sessions.json"
    if sessions_file.exists():
        sessions_file.unlink()
        logger.info("Cleaned up prior session state")

    await session.start()
    logger.info(f"Session started. Accounts: {session._all_homes}")
    logger.info(f"Session IDs: {session._session_ids}")

    # ── TURN 1: Establish a unique fact ──────────────────────────
    unique_code = f"XYZZY-{int(time.time()) % 10000}"
    prompt1 = (
        f"Remember this secret code: {unique_code}. "
        "Just say 'Got it, I'll remember {code}' and nothing else."
    )

    logger.info(f"=== TURN 1 (establishing fact) on account: {session._current_home or 'default'} ===")
    text1 = await collect_text(session, prompt1)
    logger.info(f"Turn 1 response: {text1[:200]}")
    logger.info(f"Turn 1 account used: {session._current_home or 'default'}")
    logger.info(f"Session IDs after turn 1: {session._session_ids}")

    if not session._current_session_id:
        logger.error("No session ID after turn 1!")
        return False

    # Verify JSONL exists for the account we just used
    slug = session._project_slug()
    sid = session._current_session_id
    for home in session._all_homes:
        proj_dir = session._cc_projects_dir(home) / slug
        jsonl = proj_dir / f"{sid}.jsonl"
        label = home or "default"
        exists = jsonl.exists()
        size = jsonl.stat().st_size if exists else 0
        logger.info(f"Account {label}: JSONL {'EXISTS' if exists else 'MISSING'} ({size}B)")

    # ── FORCE ROTATION ──────────────────────────────────────────
    # Mark current account as depleted so _pick_account rotates
    logger.info("=== FORCING ACCOUNT ROTATION ===")
    turn1_home = session._current_home
    session.mark_account_depleted(turn1_home, cooldown=300)

    # ── TURN 2: Ask about the fact from the new account ─────────
    prompt2 = (
        "What was the secret code I just told you? "
        "Reply with ONLY the code, nothing else."
    )

    logger.info(f"=== TURN 2 (testing context) ===")
    text2 = await collect_text(session, prompt2)
    turn2_home = session._current_home
    logger.info(f"Turn 2 response: {text2[:200]}")
    logger.info(f"Turn 2 account used: {turn2_home or 'default'}")
    logger.info(f"Session IDs after turn 2: {session._session_ids}")

    # ── VERIFY ──────────────────────────────────────────────────
    rotated = (turn1_home != turn2_home)
    code_found = unique_code in text2

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Unique code:      {unique_code}")
    print(f"Turn 1 account:   {turn1_home or 'default'}")
    print(f"Turn 2 account:   {turn2_home or 'default'}")
    print(f"Account rotated:  {'YES' if rotated else 'NO'}")
    print(f"Code in response: {'YES' if code_found else 'NO'}")
    print(f"Turn 2 response:  {text2[:300]}")
    print()

    if not rotated:
        print("FAIL: Account did not rotate (depletion didn't work)")
        return False
    if not code_found:
        print("FAIL: Context did NOT survive account rotation")
        print("  The second account could not recall the secret code.")
        print("  This means --resume is not finding the session JSONL")
        print("  on the rotated account.")
        return False

    print("PASS: Context survived account rotation!")
    return True


if __name__ == "__main__":
    result = asyncio.run(run_test())
    sys.exit(0 if result else 1)
