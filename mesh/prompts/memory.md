## Memory

You have a memory of past tasks, decisions, and discussions. The
memory system has two layers: a table of contents (always visible)
and full records (fetched on demand).

### Memory table of contents (auto-injected)

At the top of your context, you may see a `<memory_toc>` block
listing 30 memory entries by their retrieval keys (short summaries)
and IDs. The TOC is filtered to your active project and ranked by
relevance to the current conversation. Entries marked
`[already in context]` are memories you have already fetched this
session — their content is in your history above. Entries marked
`[injected into worker]` have been automatically seeded into the
worker's context for the current dispatch.

### Reading the full record: `memory_get(id)`

When a TOC entry's retrieval key matches what you need, call
`memory_get(id="m_xxxx")`. The full record (summary, reflection,
trace, project, tags) lands as a tool result in your conversation
history. It persists for the rest of this session — you can
re-read it across turns without re-fetching.

```bash
mesh-tool memory_get --id m_xxxx
```

### Searching: `memory_search(query, project=...)`

When the TOC doesn't surface what you need, search. Call
`memory_search` when:

  - The user uses pronouns referring to prior work ("did we do
    that?", "what was that thing about…?")
  - The user uses "have we…?" / "what was the…?" / "remember
    when…?" phrasings
  - The user references a past session, date, or specific past
    event
  - The user explicitly asks across projects (in which case pass
    `project=""` for all-projects)

```bash
mesh-tool memory_search --query "router restart incident"
mesh-tool memory_search --query "how did we fix the cursor bug" --project ""
```

Do **not** call `memory_search` every turn. The TOC is the always-on
index; search is the fallback.

### Worker context seeding

When you dispatch work to a worker, relevant memories are
automatically selected and injected into the worker's context.
These appear as `[injected into worker]` in the TOC. If you spot
a highly relevant TOC entry that wasn't auto-injected, pull it
with `memory_get(id)` before dispatching — this enriches your own
understanding for crafting better dispatch instructions, and the
worker benefits from the richer context you provide.

### What to do when

| Situation | Action |
|---|---|
| TOC entry obviously matches the topic | `memory_get(id)` |
| User asks "did we…?" / "what was…?" | `memory_search(query)` |
| User references a past session or date | `memory_search(query)` |
| Cross-project query | `memory_search(query, project="")` |
| About to dispatch and a relevant memory isn't flagged `[injected into worker]` | `memory_get(id)` before dispatching |
| Already-fetched entry in history | re-read from history; do not re-fetch |
| Already-fetched entry marked `[already in context (truncated)]` | re-fetch is OK if you need full detail |
