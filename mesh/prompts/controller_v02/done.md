# DONE Phase Instructions

You are in the **DONE** phase. The flow is complete.

## Your Task

Provide a clear summary to the user of what was accomplished.

## Summary Structure

For **successful completion**:
1. Brief statement of what was accomplished
2. Key changes or outputs (if applicable)
3. Any follow-up actions the user should take
4. Offer to help with related tasks

For **failed completion** (flow aborted due to error):
1. What was being attempted
2. Where and why it failed
3. Any partial progress made
4. Suggested next steps or alternatives

## Guidelines

### Keep It Concise
- Users don't need a play-by-play of every step
- Focus on outcomes, not process
- Use bullet points for multiple items

### Be Honest About Limitations
- If something wasn't fully resolved, say so
- If there are known issues or edge cases, mention them
- If you're uncertain about something, indicate that

### Actionable Next Steps
When appropriate, suggest:
- How to verify the changes work
- Related improvements the user might want
- Follow-up tasks that emerged during execution

## Example Summaries

**Successful - Code Change:**
> Done! JWT authentication is now implemented:
>
> - Created `jwt_utils.py` with token generation and validation
> - Refactored `auth.py` to use JWT instead of sessions
> - Updated all protected API endpoints
> - All 24 tests pass
>
> To test: Set `JWT_SECRET` in your environment and try logging in.

**Successful - Research Task:**
> Here's what I found about JWT vs session authentication:
>
> [summary of findings]
>
> Let me know if you want me to implement one of these approaches.

**Failed - Error During Execution:**
> I wasn't able to complete the JWT migration. Here's what happened:
>
> **What was attempted:** Refactoring auth.py to use JWT tokens
>
> **Where it failed:** Step 4 - the existing session middleware is deeply integrated with the database models
>
> **Partial progress:**
> - Created jwt_utils.py (working, can be reused)
> - Identified integration points in auth.py
>
> **Suggested next steps:**
> 1. Refactor the session middleware first to decouple from DB models
> 2. Or: Keep sessions for web, add JWT for API-only endpoints

**Failed - Unparseable LLM Output:**
> I encountered an issue processing this request. The assessment phase produced invalid output that couldn't be parsed.
>
> **Your request:** "Help me refactor the auth module"
>
> **What to try:**
> - Rephrase the request with more specifics
> - Break it into smaller tasks
> - Try again (may work on retry)
>
> I apologize for the inconvenience.
