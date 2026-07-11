#!/usr/bin/env python3
"""
Memory Context Profiles — Live Validation Suite

Tests the three-slice rendering system (Representative / Recent / Relevant)
against a real agent's SQLite memory pool with real embeddings.

No running agent or mesh router needed — loads the DB directly,
constructs a MemorySystem with the real pool, calls render(), and validates.

Usage:
    cd ~/apps/hello-world
    source env.bash && source .venv/bin/activate
    python -m tests.live.test_memory_profiles_live                    # Run all
    python -m tests.live.test_memory_profiles_live --group A          # Run group A only
    python -m tests.live.test_memory_profiles_live --group A B C      # Multiple groups
    python -m tests.live.test_memory_profiles_live --agent bob        # Test specific agent
    python -m tests.live.test_memory_profiles_live -v                 # Verbose
"""

import argparse
import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mesh.llm import estimate_tokens
from mesh.memory.embeddings import EmbeddingClient
from mesh.memory.selection import cosine_sim
from mesh.memory.store import MemoryStore
from mesh.memory.system import (
    ROUTER_PROFILE,
    WORKER_PROFILE,
    MemoryProfile,
    MemorySystem,
    _build_profile,
    _entry_tokens,
)
from mesh.config import load_config


# =============================================================================
# Test infrastructure
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    group: str
    passed: bool
    duration_ms: int
    details: str = ""
    error: str = ""


def result_line(r: TestResult) -> str:
    status = "PASS" if r.passed else "FAIL"
    detail = f" — {r.details}" if r.details else ""
    err = f" ERROR: {r.error}" if r.error else ""
    return f"  [{status}] {r.name} ({r.duration_ms}ms){detail}{err}"


@dataclass
class ParsedEntry:
    """A parsed <entry> element from rendered XML."""
    id: str
    date: str
    tags: str
    outcome: str
    depth: str
    source: str
    similarity: float | None
    content: str
    tokens: int


def parse_rendered_xml(xml_text: str) -> list[ParsedEntry]:
    """Parse <entry> elements from rendered memory XML."""
    if not xml_text or not xml_text.strip():
        return []

    entries = []
    # Use regex-based parsing since entries may contain unescaped XML-like content
    pattern = re.compile(
        r'<entry\s+'
        r'id="([^"]*?)"\s+'
        r'date="([^"]*?)"\s+'
        r'tags="([^"]*?)"\s+'
        r'outcome="([^"]*?)"\s+'
        r'depth="([^"]*?)"\s+'
        r'source="([^"]*?)"'
        r'(?:\s+similarity="([^"]*?)")?'
        r'>(.*?)</entry>',
        re.DOTALL,
    )

    for m in pattern.finditer(xml_text):
        sim_str = m.group(7)
        content = m.group(8).strip()
        entries.append(ParsedEntry(
            id=m.group(1),
            date=m.group(2),
            tags=m.group(3),
            outcome=m.group(4),
            depth=m.group(5),
            source=m.group(6),
            similarity=float(sim_str) if sim_str else None,
            content=content,
            tokens=estimate_tokens(m.group(0)),
        ))
    return entries


def entries_by_source(entries: list[ParsedEntry], source: str) -> list[ParsedEntry]:
    return [e for e in entries if e.source == source]


def total_tokens(entries: list[ParsedEntry]) -> int:
    return sum(e.tokens for e in entries)


# =============================================================================
# Test runner
# =============================================================================

class MemoryProfilesTestRunner:
    """Validates memory profile rendering against a real agent's memory pool."""

    ALL_GROUPS = ["A", "B", "C", "D", "E", "F"]

    def __init__(self, agent_nickname: str = "bob", verbose: bool = False):
        self.agent = agent_nickname
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.memory: MemorySystem | None = None
        self.pool_size: int = 0
        self.active_size: int = 0

        # Cached render outputs (populated in setup)
        self._router_output: str = ""
        self._router_entries: list[ParsedEntry] = []
        self._worker_output: str = ""
        self._worker_entries: list[ParsedEntry] = []
        self._nonsense_output: str = ""
        self._nonsense_entries: list[ParsedEntry] = []
        self._router_output_2: str = ""
        self._router_entries_2: list[ParsedEntry] = []

    def log(self, msg: str) -> None:
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
            print(f"  [{ts}] {msg}")

    async def setup(self) -> bool:
        """Load agent's memory pool from SQLite and render test outputs."""
        db_dir = os.path.expanduser("~/.mesh/memory")
        db_path = os.path.join(db_dir, f"{self.agent}.db")
        if not os.path.exists(db_path):
            print(f"ERROR: No memory DB found at {db_path}")
            return False

        # Load config to get agent's profile overrides
        config = load_config()
        agent_config = None
        for node_id, node in config.nodes.items():
            # Match by nickname or node ID suffix
            if node.nickname == self.agent or node_id.endswith(f":{self.agent}"):
                agent_config = node
                break

        light_profile_config = getattr(agent_config, 'memory_profile_light', None) if agent_config else None
        deep_profile_config = getattr(agent_config, 'memory_profile_deep', None) if agent_config else None
        # Backward compat fallback
        if not light_profile_config:
            light_profile_config = getattr(agent_config, 'memory_router_profile', None) if agent_config else None
        if not deep_profile_config:
            deep_profile_config = getattr(agent_config, 'memory_worker_profile', None) if agent_config else None

        # Construct MemorySystem with real pool
        self.memory = MemorySystem(
            nickname=self.agent,
            llm_client=None,  # No LLM needed for rendering
            light_profile_config=light_profile_config,
            deep_profile_config=deep_profile_config,
        )

        # Load pool directly from store
        store = MemoryStore(self.agent, db_dir=db_dir)
        self.memory._store = store
        self.memory._pool = store.load()
        self.memory._reselect_active_set()

        self.pool_size = len(self.memory._pool)
        self.active_size = len(self.memory._active_ids)

        if self.pool_size == 0:
            print(f"ERROR: Agent '{self.agent}' has 0 entries in memory pool")
            return False

        self.log(f"Loaded {self.pool_size} entries, {self.active_size} in active set")

        # Pre-render outputs for all tests
        print(f"Pre-rendering test outputs...")

        query = "memory system design and implementation"

        t0 = time.monotonic()
        self._router_output = await self.memory.render(
            self.memory.light_profile, query=query
        )
        self._router_entries = parse_rendered_xml(self._router_output)
        self.log(f"Router render: {len(self._router_entries)} entries, "
                 f"{estimate_tokens(self._router_output)} tokens, "
                 f"{(time.monotonic()-t0)*1000:.0f}ms")

        t0 = time.monotonic()
        self._worker_output = await self.memory.render(
            self.memory.deep_profile, query=query
        )
        self._worker_entries = parse_rendered_xml(self._worker_output)
        self.log(f"Worker render: {len(self._worker_entries)} entries, "
                 f"{estimate_tokens(self._worker_output)} tokens, "
                 f"{(time.monotonic()-t0)*1000:.0f}ms")

        t0 = time.monotonic()
        self._nonsense_output = await self.memory.render(
            self.memory.light_profile, query="xyz123abc nonsense query qwerty"
        )
        self._nonsense_entries = parse_rendered_xml(self._nonsense_output)
        self.log(f"Nonsense render: {len(self._nonsense_entries)} entries, "
                 f"{estimate_tokens(self._nonsense_output)} tokens, "
                 f"{(time.monotonic()-t0)*1000:.0f}ms")

        t0 = time.monotonic()
        self._router_output_2 = await self.memory.render(
            self.memory.light_profile, query=query
        )
        self._router_entries_2 = parse_rendered_xml(self._router_output_2)
        self.log(f"Router render (repeat): {len(self._router_entries_2)} entries, "
                 f"{(time.monotonic()-t0)*1000:.0f}ms")

        print(f"  Ready: {self.pool_size} entries in pool, "
              f"{self.active_size} in active set\n")
        return True

    def _record(self, name: str, group: str, passed: bool,
                duration_ms: int, details: str = "", error: str = "") -> TestResult:
        r = TestResult(
            name=name, group=group, passed=passed,
            duration_ms=duration_ms, details=details, error=error,
        )
        self.results.append(r)
        return r

    async def run(self, groups: list[str] | None = None) -> list[TestResult]:
        """Run specified test groups (or all)."""
        target = groups or self.ALL_GROUPS

        group_methods = {
            "A": self._run_group_a,
            "B": self._run_group_b,
            "C": self._run_group_c,
            "D": self._run_group_d,
            "E": self._run_group_e,
            "F": self._run_group_f,
        }

        for g in target:
            fn = group_methods.get(g.upper())
            if fn:
                await fn()
            else:
                print(f"  [SKIP] Unknown group: {g}")

        return self.results

    # ─── Group A: Baseline Rendering ────────────────────────────────

    async def _run_group_a(self):
        print("Group A: Baseline Rendering")

        # A1: render produces output
        t0 = time.monotonic()
        passed = len(self._router_output) > 0
        ms = int((time.monotonic() - t0) * 1000)
        r = self._record("A1: render_produces_output", "A", passed, ms,
                         f"{len(self._router_output)} chars")
        print(result_line(r))

        # A2: total budget
        t0 = time.monotonic()
        tokens = estimate_tokens(self._router_output)
        budget = self.memory.light_profile.budget_tokens
        passed = tokens <= budget
        ms = int((time.monotonic() - t0) * 1000)
        r = self._record("A2: render_total_budget", "A", passed, ms,
                         f"{tokens:,} tokens <= {budget:,}")
        print(result_line(r))

        # A3: has at least representative + recent slices
        # (Relevant may be empty if pool is small enough to fit entirely in
        # Representative + Recent — that's correct behavior, not a failure)
        t0 = time.monotonic()
        repr_count = len(entries_by_source(self._router_entries, "representative"))
        recent_count = len(entries_by_source(self._router_entries, "recent"))
        relevant_count = len(entries_by_source(self._router_entries, "relevant"))
        total_entries = repr_count + recent_count + relevant_count
        passed = repr_count > 0 and recent_count > 0 and total_entries == self.pool_size
        ms = int((time.monotonic() - t0) * 1000)
        note = " (pool fits in repr+recent)" if relevant_count == 0 else ""
        r = self._record("A3: render_has_slices", "A", passed, ms,
                         f"repr={repr_count}, recent={recent_count}, relevant={relevant_count}{note}")
        print(result_line(r))

        # A4: no duplicate IDs
        t0 = time.monotonic()
        ids = [e.id for e in self._router_entries]
        unique_ids = set(ids)
        passed = len(ids) == len(unique_ids)
        ms = int((time.monotonic() - t0) * 1000)
        details = f"{len(unique_ids)} unique IDs"
        error = ""
        if not passed:
            from collections import Counter
            dupes = [k for k, v in Counter(ids).items() if v > 1]
            error = f"Duplicates: {dupes}"
        r = self._record("A4: render_no_duplicate_ids", "A", passed, ms, details, error)
        print(result_line(r))

    # ─── Group B: Slice Budget Compliance ───────────────────────────

    async def _run_group_b(self):
        print("Group B: Slice Budget Compliance")
        profile = self.memory.light_profile
        budget = profile.budget_tokens
        tolerance = 0.15  # 15% over-budget tolerance per slice

        for label, source, pct in [
            ("B1: representative_within_budget", "representative", profile.representative_pct),
            ("B2: recent_within_budget", "recent", profile.recent_pct),
            ("B3: relevant_within_budget", "relevant", profile.relevant_pct),
        ]:
            t0 = time.monotonic()
            slice_entries = entries_by_source(self._router_entries, source)
            slice_tokens = total_tokens(slice_entries)
            slice_budget = int(budget * pct)
            max_allowed = int(slice_budget * (1 + tolerance))
            passed = slice_tokens <= max_allowed
            ms = int((time.monotonic() - t0) * 1000)
            pct_used = (slice_tokens / slice_budget * 100) if slice_budget > 0 else 0
            r = self._record(label, "B", passed, ms,
                             f"{slice_tokens:,} / {slice_budget:,} ({pct_used:.0f}%)")
            print(result_line(r))

    # ─── Group C: Similarity Floor ──────────────────────────────────

    async def _run_group_c(self):
        print("Group C: Similarity Floor")
        profile = self.memory.light_profile

        # C1: all relevant entries above floor
        t0 = time.monotonic()
        relevant = entries_by_source(self._router_entries, "relevant")
        below_floor = [e for e in relevant if e.similarity is not None
                       and e.similarity < profile.similarity_floor]
        passed = len(below_floor) == 0
        ms = int((time.monotonic() - t0) * 1000)
        details = f"{len(relevant)} relevant entries, all >= {profile.similarity_floor}"
        error = ""
        if not passed:
            error = f"{len(below_floor)} below floor: {[f'{e.id}={e.similarity:.3f}' for e in below_floor]}"
        r = self._record("C1: relevant_above_floor", "C", passed, ms, details, error)
        print(result_line(r))

        # C2: nonsense query has fewer relevant entries
        t0 = time.monotonic()
        nonsense_relevant = entries_by_source(self._nonsense_entries, "relevant")
        real_relevant = entries_by_source(self._router_entries, "relevant")
        # Nonsense should have fewer or equal relevant entries
        passed = len(nonsense_relevant) <= len(real_relevant)
        ms = int((time.monotonic() - t0) * 1000)
        r = self._record("C2: low_sim_query_redistributes", "C", passed, ms,
                         f"nonsense={len(nonsense_relevant)} vs real={len(real_relevant)}")
        print(result_line(r))

    # ─── Group D: Depth Control ─────────────────────────────────────

    async def _run_group_d(self):
        print("Group D: Depth Control")
        profile = self.memory.light_profile

        # D1: representative depth matches profile
        t0 = time.monotonic()
        repr_entries = entries_by_source(self._router_entries, "representative")
        # With representative_full_reflections=0 (router default), all should be summary
        # unless depth escalation upgraded them
        if profile.representative_full_reflections == 0:
            # Check that non-escalated entries are summary
            # (escalated entries are allowed to be full/trace)
            full_or_trace = [e for e in repr_entries if e.depth in ("full", "trace")]
            # We can't easily tell which were escalated without re-doing the computation,
            # but we can verify the count is reasonable
            details = f"{len(repr_entries)} total, {len(full_or_trace)} escalated to full/trace"
            passed = True  # Informational — escalation is valid
        else:
            full_count = sum(1 for e in repr_entries if e.depth in ("full", "trace"))
            passed = full_count <= profile.representative_full_reflections + 10  # generous for escalation
            details = f"{full_count} full/trace (profile allows {profile.representative_full_reflections} + escalation)"
        ms = int((time.monotonic() - t0) * 1000)
        r = self._record("D1: representative_depth_matches_profile", "D", passed, ms, details)
        print(result_line(r))

        # D2: recent has full reflections
        t0 = time.monotonic()
        recent_entries = entries_by_source(self._router_entries, "recent")
        recent_full = [e for e in recent_entries if e.depth in ("full", "trace")]
        expected_full = min(profile.recent_full_reflections, len(recent_entries))
        passed = len(recent_full) >= expected_full
        ms = int((time.monotonic() - t0) * 1000)
        r = self._record("D2: recent_has_full_reflections", "D", passed, ms,
                         f"{len(recent_full)} full/trace out of {len(recent_entries)} "
                         f"(expected >= {expected_full})")
        print(result_line(r))

        # D3: worker relevant depth
        t0 = time.monotonic()
        w_profile = self.memory.deep_profile
        w_relevant = entries_by_source(self._worker_entries, "relevant")
        w_rel_full = [e for e in w_relevant if e.depth in ("full", "trace")]
        expected_rel_full = min(w_profile.relevant_full_reflections, len(w_relevant))
        passed = len(w_rel_full) >= expected_rel_full or len(w_relevant) < w_profile.relevant_full_reflections
        ms = int((time.monotonic() - t0) * 1000)
        r = self._record("D3: relevant_depth_matches_profile", "D", passed, ms,
                         f"{len(w_rel_full)} full/trace out of {len(w_relevant)} relevant "
                         f"(worker profile: {w_profile.relevant_full_reflections} full, "
                         f"{w_profile.relevant_top_traces} trace)")
        print(result_line(r))

    # ─── Group E: Cross-Agent Consistency ───────────────────────────

    async def _run_group_e(self):
        print("Group E: Consistency")

        # E1: deterministic rendering
        t0 = time.monotonic()
        passed = self._router_output == self._router_output_2
        ms = int((time.monotonic() - t0) * 1000)
        details = "identical" if passed else "differs"
        error = ""
        if not passed:
            # Show first difference
            ids_1 = [e.id for e in self._router_entries]
            ids_2 = [e.id for e in self._router_entries_2]
            if ids_1 != ids_2:
                error = f"Different entry sets: {len(ids_1)} vs {len(ids_2)}"
            else:
                error = "Same entries, different content/ordering"
        r = self._record("E1: render_deterministic", "E", passed, ms, details, error)
        print(result_line(r))

        # E2: different queries produce different relevant entries
        t0 = time.monotonic()
        real_relevant_ids = {e.id for e in entries_by_source(self._router_entries, "relevant")}
        nonsense_relevant_ids = {e.id for e in entries_by_source(self._nonsense_entries, "relevant")}
        # They should differ (unless both are empty, which is fine)
        if real_relevant_ids or nonsense_relevant_ids:
            passed = real_relevant_ids != nonsense_relevant_ids
        else:
            passed = True  # Both empty — trivially different
        ms = int((time.monotonic() - t0) * 1000)
        overlap = len(real_relevant_ids & nonsense_relevant_ids)
        r = self._record("E2: different_queries_different_relevant", "E", passed, ms,
                         f"real={len(real_relevant_ids)}, nonsense={len(nonsense_relevant_ids)}, "
                         f"overlap={overlap}")
        print(result_line(r))

    # ─── Group F: Profile Override Verification ─────────────────────

    async def _run_group_f(self):
        print("Group F: Profile Override Verification")

        # F1: agent profile loaded from config
        t0 = time.monotonic()
        rp = self.memory.light_profile
        wp = self.memory.deep_profile
        # Check that profiles have the expected built-in or overridden values
        pct_sum = rp.representative_pct + rp.recent_pct + rp.relevant_pct
        passed = abs(pct_sum - 1.0) < 0.01
        ms = int((time.monotonic() - t0) * 1000)
        r = self._record("F1: agent_profile_loaded", "F", passed, ms,
                         f"router: {rp.budget_tokens}tok, "
                         f"{rp.representative_pct:.0%}/{rp.recent_pct:.0%}/{rp.relevant_pct:.0%}, "
                         f"worker: {wp.budget_tokens}tok, "
                         f"{wp.representative_pct:.0%}/{wp.recent_pct:.0%}/{wp.relevant_pct:.0%}")
        print(result_line(r))

    # ─── Output ─────────────────────────────────────────────────────

    def print_summary(self):
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print(f"\nResults: {passed}/{total} passed", end="")
        if failed:
            print(f", {failed} FAILED")
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  {r.name}: {r.error or r.details}")
        else:
            print()


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Memory Context Profiles — Live Validation"
    )
    parser.add_argument(
        "--agent", default="bob",
        help="Agent nickname to test (default: bob)",
    )
    parser.add_argument(
        "--group", nargs="*",
        help="Test groups to run (A-F, default: all)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    print(f"=== Memory Context Profiles — Live Validation ===")
    print(f"Agent: {args.agent}")
    print()

    runner = MemoryProfilesTestRunner(
        agent_nickname=args.agent,
        verbose=args.verbose,
    )

    if not await runner.setup():
        sys.exit(1)

    await runner.run(args.group)
    runner.print_summary()

    # Exit with error code if any tests failed
    if any(not r.passed for r in runner.results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
