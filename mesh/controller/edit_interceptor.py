"""
Edit interceptor - captures file write/edit operations for approval.

Detects file modification tool calls and creates EditProposals for user review.
"""

import difflib
import logging
from typing import Any
from datetime import datetime

from .models import EditProposal

logger = logging.getLogger(__name__)

# Tool names that modify files
FILE_WRITE_TOOLS = {"file_write", "file_create", "file_edit"}


class EditInterceptor:
    """
    Intercepts file write/edit operations and creates proposals for approval.

    Works with TaskFSMController to:
    - Detect file modification tool calls
    - Create EditProposal objects with diffs
    - Return proposals for task management
    """

    def __init__(self):
        """Initialize the edit interceptor."""
        pass

    def detect_file_writes(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """
        Detect file write operations from tool calls.

        Args:
            tool_calls: List of tool calls from LLM

        Returns:
            List of file write metadata dicts with:
            - tool_name: Name of the write tool
            - file_path: Target file path
            - content: New file content (or edit details)
            - arguments: Full tool arguments
        """
        file_writes = []

        for call in tool_calls:
            if not hasattr(call, 'name'):
                continue

            if call.name in FILE_WRITE_TOOLS:
                args = call.arguments if hasattr(call, 'arguments') else {}

                file_writes.append({
                    "tool_name": call.name,
                    "file_path": args.get("path", args.get("file_path", "unknown")),
                    "content": args.get("content", args.get("new_content", "")),
                    "arguments": args,
                    "call_object": call,  # Keep reference for execution
                })

        return file_writes

    def create_proposal(
        self,
        task_id: str,
        file_path: str,
        tool_name: str,
        arguments: dict[str, Any],
        old_content: str = "",
    ) -> EditProposal:
        """
        Create an EditProposal for a file write operation.

        Args:
            task_id: ID of the task this edit belongs to
            file_path: Path to the file being edited
            tool_name: Name of the tool (file_write, file_create, file_edit)
            arguments: Tool arguments
            old_content: Current file content (if file exists)

        Returns:
            EditProposal ready for user approval
        """
        # Extract new content based on tool type
        if tool_name == "file_edit":
            # For file_edit, we have old_string -> new_string
            old_str = arguments.get("old_string", "")
            new_str = arguments.get("new_string", "")
            # Apply the edit to old_content to get new_content
            new_content = old_content.replace(old_str, new_str, 1)
            description = f"Replace text in {file_path}"
        else:
            # For file_write/file_create, we have full new content
            new_content = arguments.get("content", arguments.get("new_content", ""))
            if tool_name == "file_create":
                description = f"Create new file {file_path}"
            else:
                description = f"Write to {file_path}"

        # Generate diff
        diff = self._generate_diff(
            old_content,
            new_content,
            file_path,
        )

        # Create proposal
        now = datetime.utcnow().isoformat() + "Z"
        proposal_id = f"edit-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        return EditProposal(
            id=proposal_id,
            task_id=task_id,
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            diff=diff,
            description=description,
            created_at=now,
        )

    def _generate_diff(
        self,
        old_content: str,
        new_content: str,
        file_path: str,
    ) -> str:
        """
        Generate a unified diff between old and new content.

        Args:
            old_content: Original file content
            new_content: Proposed new content
            file_path: File path (for diff header)

        Returns:
            Unified diff string
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff_lines = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )

        return "".join(diff_lines)

    def format_proposal_for_display(self, proposal: EditProposal) -> str:
        """
        Format an EditProposal for display to the user.

        Args:
            proposal: The edit proposal to format

        Returns:
            Human-readable markdown string
        """
        lines = [
            f"## Edit Proposal: {proposal.id}",
            f"**File**: `{proposal.file_path}`",
            f"**Description**: {proposal.description}",
            "",
            "**Diff**:",
            "```diff",
            proposal.diff,
            "```",
            "",
            f"Use `/approve` to apply this edit, or `/reject` to cancel.",
        ]

        return "\n".join(lines)
