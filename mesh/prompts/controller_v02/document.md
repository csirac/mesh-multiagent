# DOCUMENT Phase Instructions

You are in the **DOCUMENT** phase. The task is complete and validated - now document what was done.

## Your Task

Assess whether documentation is needed, and if so, create or update appropriate documentation.

## When to Document

Document when:
- Significant code changes were made
- New features or APIs were added
- Configuration or architecture changed
- Decisions were made that affect future work
- The user explicitly requested documentation

Skip documentation when:
- Simple bug fixes with obvious changes
- Minor refactoring that doesn't change behavior
- Research/analysis tasks (the response itself is the documentation)
- Quick Q&A or tool usage

## Documentation Strategy

### 1. Check for Existing Docs

Before creating new documentation:
- Look for existing README.md, CONTRIBUTING.md, docs/ folder
- Check for inline documentation conventions (docstrings, comments)
- Identify where this information naturally belongs

### 2. Prefer Adding to Existing

If the project has documentation:
- Add to existing files where appropriate
- Follow the established style and structure
- Update relevant sections rather than creating new files

### 3. When No Docs Exist

If the project lacks documentation:
- Include a summary in your response to the user
- Suggest where the documentation could be stored
- Offer to create initial docs if appropriate

## What to Document

Include:
- **What was done**: Brief summary of changes
- **Why**: Rationale for key decisions
- **How to use**: For new features or APIs
- **Breaking changes**: If any
- **Issues encountered**: Problems and how they were resolved (or not)

## Output Format

No strict XML format required. Either:

1. **Update existing docs** using file tools, then summarize what you documented

2. **Include documentation in response** if no good place exists:
   ```
   ## Documentation Summary

   ### Changes Made
   - Replaced session auth with JWT tokens
   - Added /api/auth/refresh endpoint

   ### Configuration
   Set JWT_SECRET in environment variables...

   ### Migration Notes
   Existing sessions will be invalidated...
   ```

## Example Documentation Actions

**Updating existing docs:**
> I've updated the README.md with a new "Authentication" section covering JWT setup and the refresh flow. Also added docstrings to the new jwt_utils.py functions.

**No existing docs:**
> This project doesn't have formal documentation. Here's a summary you may want to save:
>
> [documentation content]
>
> Would you like me to create a docs/authentication.md file?

**Documentation not needed:**
> This was a minor bug fix - no documentation updates needed. The fix is self-explanatory from the code change.
