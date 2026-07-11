"""
Unit tests for mesh/controller/edit_interceptor.py
"""

import pytest
from mesh.controller.edit_interceptor import EditInterceptor
from mesh.controller.models import EditProposal


class MockToolCall:
    """Mock tool call for testing."""
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class TestEditInterceptor:
    """Test edit interception logic."""

    @pytest.fixture
    def interceptor(self):
        """Create edit interceptor."""
        return EditInterceptor()

    def test_detect_no_file_writes(self, interceptor):
        """Test detection with no file writes."""
        calls = [
            MockToolCall("bash_exec", {"command": "ls"}),
            MockToolCall("file_read", {"path": "/test.py"}),
        ]

        detected = interceptor.detect_file_writes(calls)
        assert len(detected) == 0

    def test_detect_file_write(self, interceptor):
        """Test detection of file_write tool."""
        calls = [
            MockToolCall("file_write", {
                "path": "/tmp/test.py",
                "content": "print('hello')",
            }),
        ]

        detected = interceptor.detect_file_writes(calls)
        assert len(detected) == 1
        assert detected[0]["tool_name"] == "file_write"
        assert detected[0]["file_path"] == "/tmp/test.py"
        assert detected[0]["content"] == "print('hello')"

    def test_detect_file_create(self, interceptor):
        """Test detection of file_create tool."""
        calls = [
            MockToolCall("file_create", {
                "path": "/tmp/new.py",
                "content": "# new file",
            }),
        ]

        detected = interceptor.detect_file_writes(calls)
        assert len(detected) == 1
        assert detected[0]["tool_name"] == "file_create"

    def test_detect_file_edit(self, interceptor):
        """Test detection of file_edit tool."""
        calls = [
            MockToolCall("file_edit", {
                "path": "/tmp/existing.py",
                "old_string": "old",
                "new_string": "new",
            }),
        ]

        detected = interceptor.detect_file_writes(calls)
        assert len(detected) == 1
        assert detected[0]["tool_name"] == "file_edit"

    def test_detect_multiple_writes(self, interceptor):
        """Test detection of multiple file writes."""
        calls = [
            MockToolCall("file_write", {"path": "/tmp/a.py", "content": "a"}),
            MockToolCall("bash_exec", {"command": "ls"}),
            MockToolCall("file_create", {"path": "/tmp/b.py", "content": "b"}),
        ]

        detected = interceptor.detect_file_writes(calls)
        assert len(detected) == 2
        assert detected[0]["file_path"] == "/tmp/a.py"
        assert detected[1]["file_path"] == "/tmp/b.py"

    def test_create_proposal_file_write(self, interceptor):
        """Test proposal creation for file_write."""
        proposal = interceptor.create_proposal(
            task_id="task-001",
            file_path="/tmp/test.py",
            tool_name="file_write",
            arguments={"content": "new content"},
            old_content="old content",
        )

        assert proposal.task_id == "task-001"
        assert proposal.file_path == "/tmp/test.py"
        assert proposal.old_content == "old content"
        assert proposal.new_content == "new content"
        assert "Write to" in proposal.description
        assert proposal.diff  # Should have a diff
        assert proposal.approved is None  # Pending

    def test_create_proposal_file_create(self, interceptor):
        """Test proposal creation for file_create."""
        proposal = interceptor.create_proposal(
            task_id="task-002",
            file_path="/tmp/new.py",
            tool_name="file_create",
            arguments={"content": "# new file\nprint('hello')"},
            old_content="",
        )

        assert proposal.old_content == ""
        assert proposal.new_content == "# new file\nprint('hello')"
        assert "Create new file" in proposal.description

    def test_create_proposal_file_edit(self, interceptor):
        """Test proposal creation for file_edit."""
        old_content = "def foo():\n    return 'old'\n"
        proposal = interceptor.create_proposal(
            task_id="task-003",
            file_path="/tmp/edit.py",
            tool_name="file_edit",
            arguments={
                "old_string": "return 'old'",
                "new_string": "return 'new'",
            },
            old_content=old_content,
        )

        assert proposal.old_content == old_content
        assert "return 'new'" in proposal.new_content
        assert "Replace text" in proposal.description

    def test_generate_diff(self, interceptor):
        """Test diff generation."""
        old = "line 1\nline 2\nline 3\n"
        new = "line 1\nmodified line 2\nline 3\n"

        diff = interceptor._generate_diff(old, new, "test.py")

        assert "---" in diff
        assert "+++" in diff
        assert "line 1" in diff
        assert "+modified line 2" in diff
        assert "-line 2" in diff

    def test_format_proposal_for_display(self, interceptor):
        """Test proposal formatting for user display."""
        proposal = EditProposal(
            id="edit-001",
            task_id="task-001",
            file_path="/tmp/test.py",
            old_content="old",
            new_content="new",
            diff="--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new",
            description="Test edit",
        )

        formatted = interceptor.format_proposal_for_display(proposal)

        assert "## Edit Proposal:" in formatted
        assert "edit-001" in formatted
        assert "/tmp/test.py" in formatted
        assert "Test edit" in formatted
        assert "```diff" in formatted
        assert "/approve" in formatted
        assert "/reject" in formatted


class TestEditProposal:
    """Test EditProposal model."""

    def test_to_dict(self):
        """Test serialization."""
        proposal = EditProposal(
            id="edit-001",
            task_id="task-001",
            file_path="/tmp/test.py",
            old_content="old",
            new_content="new",
            diff="...",
            description="Test",
        )

        d = proposal.to_dict()
        assert d["id"] == "edit-001"
        assert d["task_id"] == "task-001"
        assert d["file_path"] == "/tmp/test.py"
        assert d["approved"] is None

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "edit-002",
            "task_id": "task-002",
            "file_path": "/tmp/other.py",
            "old_content": "a",
            "new_content": "b",
            "diff": "...",
            "description": "Another edit",
            "created_at": "2026-02-04T00:00:00Z",
            "approved": True,
            "approved_at": "2026-02-04T00:01:00Z",
        }

        proposal = EditProposal.from_dict(data)
        assert proposal.id == "edit-002"
        assert proposal.file_path == "/tmp/other.py"
        assert proposal.approved is True

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = EditProposal(
            id="edit-003",
            task_id="task-003",
            file_path="/tmp/roundtrip.py",
            old_content="before",
            new_content="after",
            diff="diff content",
            description="Roundtrip test",
        )

        data = original.to_dict()
        restored = EditProposal.from_dict(data)

        assert restored.id == original.id
        assert restored.file_path == original.file_path
        assert restored.old_content == original.old_content
        assert restored.new_content == original.new_content
