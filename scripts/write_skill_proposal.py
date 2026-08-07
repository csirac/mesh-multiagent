#!/usr/bin/env python3
"""Validate and atomically stage one governed skill-card proposal.

Drafting workers write YAML to an isolated staging path, then invoke this
command.  Invalid cards fail before anything is created under ``.proposals``.
The command has deliberately no activation or index-writing capability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh.procedural_memory import SkillCardError, SkillStore  # noqa: E402


def _summary(path: Path) -> dict[str, object]:
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "status": "proposed",
        "proposal_path": str(path),
        "card_id": card["id"],
        "purpose": card["purpose"],
        "preconditions": card["preconditions"],
        "required_invariants": card["required_invariants"],
        "verification": card["verification"],
        "caveats": card["caveats"],
        "human_approval_required": True,
        "index_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="Per-agent skill-store owner.")
    parser.add_argument(
        "--draft",
        required=True,
        type=Path,
        help="Staged YAML draft. This file is never used directly for retrieval.",
    )
    parser.add_argument(
        "--source-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "Authoritative local source included in the card. Repeat for every "
            "named local source; each fingerprint is verified."
        ),
    )
    parser.add_argument(
        "--skills-root",
        type=Path,
        help="Override the skill root for tests. Production defaults to ~/.mesh/skills.",
    )
    args = parser.parse_args()

    try:
        store = SkillStore(args.owner, root=args.skills_root)
        proposal_path = store.write_proposal_file(
            args.draft,
            source_files=args.source_file,
        )
    except (OSError, yaml.YAMLError, SkillCardError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}))
        return 1

    print(json.dumps(_summary(proposal_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
