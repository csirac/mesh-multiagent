# PLAN Phase Instructions

You are in the **PLAN** phase. The INFO phase determined this task requires planning before execution.

## Your Task

Create a concrete execution plan with clear steps. The plan should be actionable and verifiable.

## Output Format

Provide your plan in the following XML format:

```xml
<plan>
  <quality>0.0-1.0</quality>
  <steps>
    <step>
      <number>1</number>
      <description>Clear, actionable description of what to do</description>
      <expected_outcome>What success looks like for this step</expected_outcome>
    </step>
    <step>
      <number>2</number>
      <description>...</description>
      <expected_outcome>...</expected_outcome>
    </step>
  </steps>
  <rollback>How to undo changes if something goes wrong (if applicable)</rollback>
</plan>
```

## Guidelines

### Step Design

1. **Maximum 7 steps** - If you need more, combine related actions or reconsider scope
2. **Each step should be completable** - Avoid vague steps like "implement the feature"
3. **Steps are high-level** - Each step may require multiple LLM turns or tool calls
4. **Expected outcomes are verifiable** - How will you know the step succeeded?

### Quality Self-Assessment

Score your plan quality (0-1):
- **0.9-1.0**: Comprehensive, clear steps with verifiable outcomes
- **0.7-0.9**: Good plan with minor gaps
- **0.5-0.7**: Workable but missing details or unclear outcomes
- **Below 0.5**: Plan needs significant revision

If your quality score is below 0.8 for a high-complexity task, revise the plan before proceeding.

### Rollback Strategy

Include a rollback strategy when:
- Making file changes that could break things
- Modifying configuration or state
- Operations that are hard to undo manually

Skip rollback for:
- Pure research/analysis tasks
- Read-only operations
- Tasks where rollback is trivial

## Complexity Reassessment

After planning, you may realize the task is simpler or more complex than initially assessed. If so, note this:

```xml
<complexity_update>
  <original>0.75</original>
  <revised>0.45</revised>
  <reason>Task only requires a single file change, not full module refactor</reason>
</complexity_update>
```

This allows skipping VALIDATE/DOCUMENT for tasks that turn out simple.

## Example Plan

```xml
<plan>
  <quality>0.85</quality>
  <steps>
    <step>
      <number>1</number>
      <description>Read current auth.py to understand existing session-based authentication</description>
      <expected_outcome>Understanding of current auth flow, session storage, and middleware hooks</expected_outcome>
    </step>
    <step>
      <number>2</number>
      <description>Install PyJWT library and add to requirements.txt</description>
      <expected_outcome>Library installed, requirements updated</expected_outcome>
    </step>
    <step>
      <number>3</number>
      <description>Create jwt_utils.py with token generation and validation functions</description>
      <expected_outcome>Working JWT utilities with proper secret handling</expected_outcome>
    </step>
    <step>
      <number>4</number>
      <description>Refactor auth.py to use JWT instead of sessions</description>
      <expected_outcome>Auth module uses JWT, old session code removed</expected_outcome>
    </step>
    <step>
      <number>5</number>
      <description>Update API endpoints to expect Authorization header</description>
      <expected_outcome>All protected endpoints validate JWT</expected_outcome>
    </step>
    <step>
      <number>6</number>
      <description>Run existing tests and fix any failures</description>
      <expected_outcome>All tests pass with new auth system</expected_outcome>
    </step>
  </steps>
  <rollback>Git stash or revert commits to restore session-based auth</rollback>
</plan>
```
