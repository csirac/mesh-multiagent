# Task Controller System

The controller framework provides intelligent task management for mesh agents. It automatically tracks tasks, manages plan steps, and ensures safe file operations through an approval workflow.

## Overview

```
┌─────────────────────────────────────────────────────┐
│                    AgentNode                         │
│              (integration point)                     │
├─────────────────────────────────────────────────────┤
│                  Controller                          │
│  ┌──────────────┬──────────────┬──────────────────┐ │
│  │  RouterLLM   │PhaseDetector │ EditInterceptor  │ │
│  │ (classify)   │(transitions) │   (approval)     │ │
│  └──────────────┴──────────────┴──────────────────┘ │
├─────────────────────────────────────────────────────┤
│                  Persistence                         │
│               (~/log/assistant/)                     │
└─────────────────────────────────────────────────────┘
```

## Controller Modes

### Passthrough (Default)

No task tracking. Messages flow directly to the LLM. This preserves backward compatibility.

```yaml
nodes:
  agent:assistant:alice:
    controller:
      mode: passthrough  # Or omit controller section entirely
```

### Task FSM v0

Enables full task management with:
- Automatic task creation from user requests
- Phase transitions based on LLM output
- File edit approval workflow
- Task/plan/step management commands

```yaml
nodes:
  agent:assistant:alice:
    controller:
      mode: task-fsm-v0
      tasks_path: ~/log/assistant/tasks.json
      router_model: gpt-4o-mini
      router_backend: openai
```

## Task Lifecycle

### Phases

| Phase | Description |
|-------|-------------|
| `planning` | Initial phase - gathering requirements, creating plan |
| `executing` | Implementing the solution |
| `reviewing` | Checking results, running tests |
| `waiting_approval` | Pending user approval for file writes |
| `blocked` | Cannot proceed without user input |
| `done` | Task completed |

### Automatic Phase Transitions

The controller analyzes LLM output to detect phase changes:

| Signal | Transition |
|--------|------------|
| File write tool called | → `waiting_approval` |
| "task complete", "all done" | → `done` |
| "I need clarification", "please clarify" | → `blocked` |
| "waiting for approval" | → `blocked` |

### Router Classification

The RouterLLM classifies incoming messages:

| Action | Description |
|--------|-------------|
| `CREATE_TASK` | User wants help with a multi-step task |
| `ROUTE_TO_TASK` | Message relates to active task |
| `DIRECT_ANSWER` | Simple question, no task needed |

## Edit Approval Workflow

When the LLM wants to write/create/edit files:

1. **Intercept**: File operations are captured, not executed
2. **Prompt**: User sees "N edit(s) require approval"
3. **Review**: User can `/diff` to see changes
4. **Decide**: `/approve` to apply, `/reject` to cancel
5. **Continue**: Task resumes in `executing` phase

This ensures the user always controls what changes are made to their filesystem.

## Persistence

Tasks are saved to `~/log/assistant/<nickname>-tasks.json` with:
- Atomic writes (temp file + rename)
- Automatic backup before overwrite
- Recovery from backup on parse failure

State is loaded on agent startup and saved on shutdown.

## Plan Steps

Each task can have plan steps with status tracking:

| Status | Icon | Description |
|--------|------|-------------|
| `pending` | `○` | Not started |
| `in_progress` | `◐` | Currently working on |
| `completed` | `●` | Done |
| `blocked` | `⊗` | Cannot proceed |
| `skipped` | `⊘` | Intentionally skipped |

## Configuration Reference

```yaml
nodes:
  agent:assistant:alice:
    llm_backend: "openai-reasoning-medium"
    llm_model: "gpt-5.1"
    system_prompt_file: "assistant.md"
    tools: *all_tools

    controller:
      # Controller mode: "passthrough" or "task-fsm-v0"
      mode: task-fsm-v0

      # Path to persist tasks (default: ~/log/assistant/<nickname>-tasks.json)
      tasks_path: ~/log/assistant/alice-tasks.json

      # Model for router classification (default: gpt-4o-mini)
      router_model: gpt-4o-mini

      # Backend for router calls (default: openai)
      router_backend: openai
```

## See Also

- [Commands Reference](commands.md) - All task/plan/step commands
- [README](../README.md) - General mesh documentation
