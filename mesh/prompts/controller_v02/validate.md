# VALIDATE Phase Instructions

You are in the **VALIDATE** phase. Execution is complete - now verify the results.

## Your Task

Self-check whether the original task was accomplished correctly. Provide a structured assessment.

## Output Format

Provide your validation in the following XML format:

```xml
<validation>
  <task_accomplished>0.0-1.0</task_accomplished>
  <verified>0.0-1.0</verified>
  <issues>
    <issue>Description of any problem found</issue>
    <issue>Another issue if applicable</issue>
  </issues>
  <can_fix_without_replan>true/false</can_fix_without_replan>
</validation>
```

### Score Meanings

| Field | Score | Meaning |
|-------|-------|---------|
| `task_accomplished` | 0.9-1.0 | Task fully completed as requested |
| `task_accomplished` | 0.7-0.9 | Mostly complete, minor gaps |
| `task_accomplished` | 0.5-0.7 | Partially complete, significant gaps |
| `task_accomplished` | Below 0.5 | Task not completed or major issues |
| `verified` | 0.9-1.0 | Thoroughly verified (tests pass, output checked) |
| `verified` | 0.7-0.9 | Verified key functionality |
| `verified` | Below 0.7 | Limited or no verification possible |

## Guidelines

### Verification Methods

Use appropriate verification for the task type:

| Task Type | Verification |
|-----------|--------------|
| Code changes | Run tests, check syntax, verify imports |
| File creation | Verify file exists, content is correct |
| Configuration | Test that config is applied |
| Research/analysis | Cross-check facts, verify sources |
| Communication | Confirm message was sent/drafted correctly |

### Issue Handling

If you find issues:

1. **Assess fixability**: Can you fix this without replanning the whole task?
   - Minor bugs, typos, missed edge cases → `can_fix_without_replan: true`
   - Fundamental approach wrong, scope creep → `can_fix_without_replan: false`

2. **If fixable**: Attempt the fix directly in this phase, then re-verify

3. **If not fixable**: Report issues clearly to the user - no looping back to PLAN

### Pass/Fail Thresholds

Validation passes if:
- `task_accomplished >= 0.8` AND
- `verified >= 0.7`

If below thresholds with unfixable issues:
- Report issues clearly to the user
- Proceed to DONE (no retry loops)
- DOCUMENT phase only runs if validation passes

## Example Validations

**Successful validation:**
```xml
<validation>
  <task_accomplished>0.95</task_accomplished>
  <verified>0.90</verified>
  <issues></issues>
  <can_fix_without_replan>false</can_fix_without_replan>
</validation>
```
All tests pass, JWT auth working correctly.

**Validation with fixable issue:**
```xml
<validation>
  <task_accomplished>0.80</task_accomplished>
  <verified>0.85</verified>
  <issues>
    <issue>Missing token refresh endpoint</issue>
  </issues>
  <can_fix_without_replan>true</can_fix_without_replan>
</validation>
```
Main functionality works, but I can add the refresh endpoint without replanning.

**Validation with unfixable issues:**
```xml
<validation>
  <task_accomplished>0.50</task_accomplished>
  <verified>0.70</verified>
  <issues>
    <issue>JWT approach conflicts with existing session middleware</issue>
    <issue>Would require architectural changes beyond original scope</issue>
  </issues>
  <can_fix_without_replan>false</can_fix_without_replan>
</validation>
```
Reporting this to user - the approach needs reconsideration.
