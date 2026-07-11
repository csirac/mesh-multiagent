"""
phase_complete tool — lets the executor signal it has finished a phase.

Raises PhaseCompleteSignal, which is caught by run_loop to break
immediately instead of looping back for another LLM turn.
"""

from __future__ import annotations

from ...tools import tool, ToolParameter


class PhaseCompleteSignal(Exception):
    """Sentinel raised by phase_complete to break the run_loop."""

    def __init__(self, summary: str) -> None:
        self.summary = summary
        super().__init__(summary)


@tool(
    name="phase_complete",
    description=(
        "Signal that you have finished all deliverables for the current phase. "
        "Call this once you are done — it ends the phase immediately. "
        "Include a brief summary of what you accomplished."
    ),
    parameters=[
        ToolParameter(
            name="summary",
            type="string",
            description="Brief summary of what was accomplished in this phase",
            required=True,
        ),
    ],
)
def phase_complete(summary: str) -> str:
    raise PhaseCompleteSignal(summary)
