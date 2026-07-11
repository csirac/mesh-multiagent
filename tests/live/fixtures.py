"""
Fixture environment management for live tests.

Handles cloning the test fixtures repo and providing isolated
working directories for each test scenario.
"""

import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

# For local testing on server: /home/git/test-fixtures.git
# For remote testing via SSH: git@your-host.example.com:/path/to/test-fixtures.git
DEFAULT_FIXTURE_REPO = os.environ.get(
    "FIXTURE_REPO",
    "/home/git/test-fixtures.git"
)
TEMP_BASE = Path("/tmp/live-tests")


@dataclass
class TestEnvironment:
    """Isolated test environment with fresh fixture clone."""
    work_dir: Path
    fixture_repo: str
    scenario_id: str
    branch: str

    @classmethod
    async def create(
        cls,
        scenario_id: str,
        fixture_repo: str = DEFAULT_FIXTURE_REPO,
        branch: str = "main",
    ) -> "TestEnvironment":
        """Clone fixture repo to temp directory."""
        # Create unique work directory
        TEMP_BASE.mkdir(parents=True, exist_ok=True)
        work_dir = TEMP_BASE / f"test-{scenario_id}-{uuid.uuid4().hex[:8]}"

        # Clone the repo
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", "--branch", branch,
            fixture_repo, str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to clone fixtures: {stderr.decode()}")

        return cls(
            work_dir=work_dir,
            fixture_repo=fixture_repo,
            scenario_id=scenario_id,
            branch=branch,
        )

    async def checkout_branch(self, branch: str) -> None:
        """Switch to a different fixture branch."""
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", branch,
            cwd=self.work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to checkout branch {branch}: {stderr.decode()}")

        self.branch = branch

    async def reset(self) -> None:
        """Reset the working directory to clean state."""
        proc = await asyncio.create_subprocess_exec(
            "git", "reset", "--hard", "HEAD",
            cwd=self.work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        proc = await asyncio.create_subprocess_exec(
            "git", "clean", "-fd",
            cwd=self.work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    def cleanup(self) -> None:
        """Delete the temp clone."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def get_sample_project_path(self) -> Path:
        """Get path to the sample_project directory."""
        return self.work_dir / "sample_project"


@asynccontextmanager
async def test_environment(
    scenario_id: str,
    fixture_repo: str = DEFAULT_FIXTURE_REPO,
    branch: str = "main",
) -> AsyncIterator[TestEnvironment]:
    """
    Context manager for test environments.

    Usage:
        async with test_environment("T1", branch="main") as env:
            # env.work_dir is the path to the fixture clone
            # run tests against env.get_sample_project_path()
    """
    env = await TestEnvironment.create(
        scenario_id=scenario_id,
        fixture_repo=fixture_repo,
        branch=branch,
    )
    try:
        yield env
    finally:
        env.cleanup()


async def verify_fixtures_available(fixture_repo: str = DEFAULT_FIXTURE_REPO) -> bool:
    """Check if the fixture repo is accessible."""
    proc = await asyncio.create_subprocess_exec(
        "git", "ls-remote", fixture_repo,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc.returncode == 0


def cleanup_stale_environments(max_age_hours: int = 24) -> int:
    """Clean up old test environments that weren't properly cleaned."""
    import time

    cleaned = 0
    if not TEMP_BASE.exists():
        return cleaned

    cutoff = time.time() - (max_age_hours * 3600)

    for path in TEMP_BASE.iterdir():
        if path.is_dir() and path.name.startswith("test-"):
            try:
                if path.stat().st_mtime < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
                    cleaned += 1
            except OSError:
                pass

    return cleaned
