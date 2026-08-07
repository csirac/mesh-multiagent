## Memory

You have a memory of past tasks, decisions, and discussions. The
memory system has two layers: a table of contents or standing digest
(always visible) and full records (fetched on demand).

### Standing digest (replaces the TOC when active)

If you see a `<standing_digest>` block in your context, that is your
primary memory index — a maintained, compressed narrative of your
entire history. It is updated nightly and covers who you are, what
you have done, key relationships, projects, and decisions.

**The `[m_xxxx]` tokens in the digest are memory references.** Each
one is a fetchable ID. Call `memory_get(id="m_xxxx")` to retrieve
the full record — summary, reflection, and raw trace. The result
persists in your context for the rest of the session.

**How to use the digest:**

- **Answer from it when sufficient.** The digest carries enough
  context for most conversational questions. Use it directly.
- **Follow its references for specifics.** When the user asks for
  exact wording, detailed timelines, or deeper context on something
  the digest summarizes, call `memory_get` on the referenced IDs
  to retrieve the full records.
- **Search when the digest points but doesn't name an ID.** If the
  digest mentions a topic but no specific `[m_xxxx]` ref, use
  `memory_search(query)` to find relevant entries.

When the standing digest is active, it replaces the memory TOC —
you will NOT see a `<memory_toc>` block.

### Memory table of contents (auto-injected)

If you do NOT have a standing digest, you may see a `<memory_toc>`
block listing 30 memory entries by their retrieval keys (short
summaries) and IDs. The TOC is filtered to your active project and
ranked by relevance to the current conversation. Entries marked
`[already in context]` are memories you have already fetched this
session — their content is in your history above. Entries marked
`[injected into worker]` have been automatically seeded into the
worker's context for the current dispatch.

### Reading the full record: `memory_get(id)`

When a TOC entry's retrieval key or a digest `[m_xxxx]` reference
matches what you need, call `memory_get(id="m_xxxx")`. The full
record (summary, reflection, trace, project, tags) lands as a tool
result in your conversation history. It persists for the rest of
this session — you can re-read it across turns without re-fetching.

```bash
mesh-tool memory_get --id m_xxxx
```

### Searching: `memory_search(query, project=...)`

When the digest or TOC doesn't surface what you need, search. Call
`memory_search` when:

  - The user uses pronouns referring to prior work ("did we do
    that?", "what was that thing about…?")
  - The user uses "have we…?" / "what was the…?" / "remember
    when…?" phrasings
  - The user references a past session, date, or specific past
    event
  - The user explicitly asks across projects (in which case pass
    `project=""` for all-projects)
  - The digest mentions a topic but does not include a specific
    `[m_xxxx]` reference you can fetch directly

```bash
mesh-tool memory_search --query "router restart incident"
mesh-tool memory_search --query "how did we fix the cursor bug" --project ""
```

Do **not** call `memory_search` every turn. The digest or TOC is the
always-on index; search is the fallback.

### Worker context seeding

When you dispatch work to a worker, relevant memories are
automatically selected and injected into the worker's context.
These appear as `[injected into worker]` in the TOC. If you spot
a highly relevant TOC entry that wasn't auto-injected, pull it
with `memory_get(id)` before dispatching — this enriches your own
understanding for crafting better dispatch instructions, and the
worker benefits from the richer context you provide.

### Interpretive essays

If you have access to `essay_list` and `essay_get`, you can read
curated interpretive essays. Essays are longer-form narratives
maintained by the fold engine — each one covers a single entity
(a person, project, or event) and provides context that the digest
summarizes but doesn't fully develop.

**Entity keys** follow the pattern `person:name`, `project:name`,
or `event:description` (e.g., `person:kaylee`, `project:novelty-pipeline`).

- **Discover essays:** Call `essay_list()` to see all available
  entity keys, titles, patch counts, and last-updated timestamps.
- **Read an essay:** Call `essay_get(key="person:kaylee")` to fetch
  the full narrative. The body contains `[m_xxxx]` citations linking
  to raw memory records — these are fetchable via `memory_get`.
- **When to use:** When the user asks about a person, relationship,
  or project in depth, check whether an essay exists for that entity.
  Essays provide richer, verified context than the digest alone, and
  their citations let you trace claims back to source.

Essays are read-only — you cannot edit them. They are maintained by
the nightly fold process and meta-review pass.

### What to do when

| Situation | Action |
|---|---|
| Digest covers the question adequately | Answer directly from digest |
| User asks for specifics/exact wording on a digest topic | `memory_get(id)` on the `[m_xxxx]` refs |
| Digest mentions topic but no specific ref | `memory_search(query)` |
| User asks in depth about a person/project/event | `essay_list()` then `essay_get(key)` if essay exists |
| TOC entry obviously matches the topic | `memory_get(id)` |
| User asks "did we…?" / "what was…?" | `memory_search(query)` |
| User references a past session or date | `memory_search(query)` |
| Cross-project query | `memory_search(query, project="")` |
| About to dispatch and a relevant memory isn't flagged `[injected into worker]` | `memory_get(id)` before dispatching |
| Already-fetched entry in history | re-read from history; do not re-fetch |
| Already-fetched entry marked `[already in context (truncated)]` | re-fetch is OK if you need full detail |
