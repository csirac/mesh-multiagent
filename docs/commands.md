# Controller Commands Reference

This document covers all commands available when using the `task-fsm-v0` controller mode.

## Task Commands

### `/tasks` - List Tasks

```
/tasks          # List active tasks
/tasks --all    # List all tasks including completed
```

Output shows task ID, title, phase, and step progress.

### `/task` - Show/Switch Task

```
/task           # Show details of active task
/task <id>      # Switch to task by ID
```

Task details include:
- Title and description
- Current phase
- Plan steps with status
- Pending edits (if any)

### `/task done` - Complete Task

```
/task done      # Mark active task as done
/task done <id> # Mark specific task as done
```

### `/task delete` - Delete Task

```
/task delete <id>   # Delete a task
```

### `/task reopen` - Reopen Completed Task

```
/task reopen <id>   # Reopen a DONE task back to planning phase
```

## Plan Commands

### `/plan` - Show Plan

```
/plan           # Show all plan steps for active task
```

Output format:
```
Plan for: task-20260204-001
1. ○ Define sorting function
2. ● Write unit tests
3. ○ Add documentation
```

Status icons: `○` pending, `◐` in progress, `●` completed, `⊗` blocked, `⊘` skipped

### `/plan add` - Add Step

```
/plan add <description>    # Add a new step to the plan
```

Example:
```
/plan add Write unit tests for edge cases
→ Added step 3: Write unit tests for edge cases
```

### `/plan edit` - Edit Step

```
/plan edit <N> <new description>   # Edit step N's description
```

Example:
```
/plan edit 2 Write comprehensive tests with edge cases
→ Updated step 2
```

### `/plan delete` - Delete Step

```
/plan delete <N>   # Remove step N from the plan
```

### `/plan reorder` - Move Step

```
/plan reorder <from> <to>   # Move step from position to new position
```

Example:
```
/plan reorder 3 1   # Move step 3 to position 1
```

## Step Commands

### `/step done` - Mark Complete

```
/step done <N>     # Mark step N as completed
```

### `/step block` - Mark Blocked

```
/step block <N>           # Mark step N as blocked
/step block <N> <reason>  # Mark blocked with reason
```

### `/step skip` - Skip Step

```
/step skip <N>     # Mark step N as skipped
```

## Edit Approval Commands

When the LLM proposes file changes, you'll see:

```
1 edit(s) require approval. Use /approve to apply, /reject to cancel, or /diff to review.
```

### `/diff` - Review Changes

```
/diff              # Show unified diff of all pending edits
```

Output format:
```
--- a/src/main.py
+++ b/src/main.py
@@ -10,3 +10,5 @@
 def hello():
     print("Hello")
+    return True
```

### `/approve` - Apply Changes

```
/approve           # Apply all pending edits
```

Files are written to disk and the task returns to `executing` phase.

### `/reject` - Cancel Changes

```
/reject            # Reject all pending edits
```

No files are modified. The task returns to `executing` phase.

## Quick Reference

| Command | Description |
|---------|-------------|
| `/tasks` | List active tasks |
| `/tasks --all` | List all tasks |
| `/task` | Show active task |
| `/task <id>` | Switch to task |
| `/task done [id]` | Mark task complete |
| `/task delete <id>` | Delete task |
| `/task reopen <id>` | Reopen completed task |
| `/plan` | Show plan steps |
| `/plan add <desc>` | Add step |
| `/plan edit <N> <desc>` | Edit step |
| `/plan delete <N>` | Delete step |
| `/plan reorder <from> <to>` | Move step |
| `/step done <N>` | Mark step complete |
| `/step block <N> [reason]` | Mark step blocked |
| `/step skip <N>` | Skip step |
| `/diff` | Review pending edits |
| `/approve` | Apply pending edits |
| `/reject` | Reject pending edits |

## See Also

- [Controller System](controller.md) - Architecture and configuration
- [README](../README.md) - General mesh documentation
