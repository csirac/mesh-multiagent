"""Essay fold integration (Phase 2, per-agent-standing-digest spec §10a–10b).

Provides the admission/promotion logic, token-budget validation, and
placeholder guard for essays edited during nightly fold rounds.  The fold
driver imports this module and calls its functions — all essay state flows
through the same isolated unified DB used for memory formation.

Entity keys follow the convention: ``person:<name>``, ``project:<name>``,
``event:<description>``, ``topic:<label>``.  When two real entities share a
surface name, qualify the key at memory-creation time, e.g.
``person:lily-austin`` and ``person:lily-work``.
"""
import json
import re
import sqlite3
from datetime import datetime, timezone

from ..procedural_memory import SkillCardError, SkillStore, detect_formation_signal


# Same regex the digest driver uses — catch any [[...]] placeholder that
# should never survive into a committed essay.
BRACKET_TOKEN_RE = re.compile(r"\[\[[^\]\r\n]+\]\]")
LOOSE_TAG_RE = re.compile(r"(?<![0-9A-Za-z_])m_([0-9a-fA-F]{1,40})(?![0-9a-fA-F])")

ESSAYS_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS essays (
    entity_key   TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    body         TEXT NOT NULL DEFAULT '',
    citations    TEXT NOT NULL DEFAULT '[]',
    cross_refs   TEXT NOT NULL DEFAULT '[]',
    patch_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    curated_version INTEGER NOT NULL DEFAULT 0,
    verified_hash TEXT NOT NULL DEFAULT '',
    verified_at TEXT
)"""


def _ensure_essays_table(con: sqlite3.Connection) -> None:
    """Create the essays table if it doesn't exist, using the canonical DDL."""
    con.execute(ESSAYS_TABLE_DDL)
    columns = {
        row[1] for row in con.execute("PRAGMA table_info(essays)").fetchall()
    }
    if "curated_version" not in columns:
        con.execute(
            "ALTER TABLE essays ADD COLUMN "
            "curated_version INTEGER NOT NULL DEFAULT 0"
        )
    if "verified_hash" not in columns:
        con.execute(
            "ALTER TABLE essays ADD COLUMN "
            "verified_hash TEXT NOT NULL DEFAULT ''"
        )
    if "verified_at" not in columns:
        con.execute("ALTER TABLE essays ADD COLUMN verified_at TEXT")
    con.commit()


# ---------------------------------------------------------------------------
# Entity recurrence tracking
# ---------------------------------------------------------------------------

def _ensure_entity_mentions_table(con: sqlite3.Connection) -> None:
    """Create the entity_mentions table if it doesn't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS entity_mentions (
            entity_key   TEXT NOT NULL,
            round_no     INTEGER NOT NULL,
            PRIMARY KEY (entity_key, round_no)
        )
    """)
    con.commit()


def record_entity_mentions(
    db_path: str,
    entities: list[str],
    round_no: int,
) -> None:
    """Record that *entities* appeared in fold round *round_no*."""
    con = sqlite3.connect(db_path)
    _ensure_entity_mentions_table(con)
    for key in entities:
        con.execute(
            "INSERT OR IGNORE INTO entity_mentions (entity_key, round_no) "
            "VALUES (?, ?)",
            (key, round_no),
        )
    con.commit()
    con.close()


def entity_window_count(db_path: str, entity_key: str) -> int:
    """Return the number of distinct fold windows in which *entity_key* has appeared."""
    con = sqlite3.connect(db_path)
    _ensure_entity_mentions_table(con)
    row = con.execute(
        "SELECT COUNT(DISTINCT round_no) FROM entity_mentions "
        "WHERE entity_key = ?",
        (entity_key,),
    ).fetchone()
    con.close()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Admission / promotion check
# ---------------------------------------------------------------------------

class EssayAction:
    """Result of the admission check for one entity in a fold round."""
    __slots__ = ("entity_key", "action", "existing_body", "existing_title",
                 "window_count")

    PATCH = "patch"
    CREATE = "create"
    SKIP = "skip"

    def __init__(self, entity_key: str, action: str,
                 existing_body: str = "", existing_title: str = "",
                 window_count: int = 0):
        self.entity_key = entity_key
        self.action = action
        self.existing_body = existing_body
        self.existing_title = existing_title
        self.window_count = window_count

    def __repr__(self) -> str:
        return f"EssayAction({self.entity_key!r}, {self.action!r}, windows={self.window_count})"


def check_admission(
    db_path: str,
    entities: list[str],
    round_no: int,
    threshold: int = 3,
    digest_text: str | None = None,
    digest_db_path: str | None = None,
) -> list[EssayAction]:
    """Run the admission/promotion check for a set of entities.

    When *digest_text* is provided (the production path), the digest is the
    authoritative gate: an entity is admissible ONLY if it appears in the
    digest via extract_digest_entities().  Entity-mentions recurrence is
    demoted to a secondary maturity filter — an entity must be both
    in-digest AND meet the recurrence threshold for CREATE.

    When *digest_text* is None, the legacy recurrence-only gate applies
    (retained for --rebuild-entity and backward-compat test paths).
    """
    record_entity_mentions(db_path, entities, round_no)

    digest_entity_keys: set[str] | None = None
    if digest_text is not None:
        digest_entity_keys = set(
            extract_digest_entities(digest_text, db_path=digest_db_path).keys()
        )

    con = sqlite3.connect(db_path)
    _ensure_entity_mentions_table(con)
    _ensure_essays_table(con)

    actions = []
    for key in entities:
        wc = con.execute(
            "SELECT COUNT(DISTINCT round_no) FROM entity_mentions "
            "WHERE entity_key = ?",
            (key,),
        ).fetchone()[0]

        if digest_entity_keys is not None and key not in digest_entity_keys:
            actions.append(EssayAction(key, EssayAction.SKIP, window_count=wc))
            continue

        row = con.execute(
            "SELECT body, title FROM essays WHERE entity_key = ?",
            (key,),
        ).fetchone()

        if row is not None:
            actions.append(EssayAction(
                key, EssayAction.PATCH,
                existing_body=row[0], existing_title=row[1],
                window_count=wc,
            ))
        elif wc >= threshold:
            actions.append(EssayAction(
                key, EssayAction.CREATE, window_count=wc,
            ))
        else:
            actions.append(EssayAction(
                key, EssayAction.SKIP, window_count=wc,
            ))

    con.close()
    return actions


# ---------------------------------------------------------------------------
# Essay validation (mirrors validate_digest)
# ---------------------------------------------------------------------------

def validate_essay(
    body: str,
    token_budget: int = 4000,
    ntokens_fn=None,
    known_ids: set[str] | None = None,
) -> str | None:
    """Validate an essay body. Returns an error string or None if valid.

    Checks:
      1. Not empty
      2. No unresolved [[...]] placeholder tokens
      3. Token budget not exceeded
      4. All [m_xxxx] citations resolve (if known_ids provided)
    """
    if not body or not body.strip():
        return "essay body is empty"

    unresolved = sorted(set(BRACKET_TOKEN_RE.findall(body)))
    if unresolved:
        return (
            f"unresolved double-bracket placeholders {unresolved} "
            "— committed essays may contain only canonical [m_<id>] "
            "memory citations"
        )

    if ntokens_fn is not None:
        tok = ntokens_fn(body)
        if tok > token_budget:
            return (
                f"essay over budget by {tok - token_budget} tokens "
                f"({tok} > {token_budget})"
            )

    if known_ids is not None:
        cited = {m.group(1) for m in LOOSE_TAG_RE.finditer(body)}
        unresolved_ids = cited - known_ids
        if unresolved_ids:
            return (
                f"unresolved memory citations: "
                f"{sorted(unresolved_ids)[:5]}"
            )

    return None


def _inline_memory_citations(body: str) -> list[str]:
    """Return canonical memory citations that actually appear in the body."""
    return list(dict.fromkeys(
        f"m_{m.group(1)}" for m in LOOSE_TAG_RE.finditer(body)
    ))


_COLLISION_MARKER_RE = re.compile(
    r"\b(?:another|different|other|second)\s+{name}\b"
    r"|\b{name}\s+\((?:not|different|other)\b",
    re.IGNORECASE,
)


def _entity_name_from_key(entity_key: str) -> str:
    """Return the human-facing entity label from an entity key."""
    return (
        entity_key.split(":", 1)[1].replace("-", " ")
        if ":" in entity_key else entity_key.replace("-", " ")
    )


def _detect_entity_collisions(db_path: str, entity_key: str) -> list[str]:
    """Heuristically flag likely same-name entity conflation.

    This is intentionally conservative: it reports only explicit signals such
    as multiple already-qualified topic labels (``lily-austin`` and
    ``lily-work``), "another Lily" language, or multiple "Name from Place"
    origins.  It is designed to miss ambiguous cases rather than warn on
    ordinary multi-faceted relationships.
    """
    entity_name = _entity_name_from_key(entity_key).strip().lower()
    if not entity_name:
        return []

    base_token = entity_name.split()[0]
    if len(base_token) < 3:
        return []

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT summary, reflection, retrieval_key, topic_label, project, "
            "tags, created_at FROM memories "
            "WHERE lower(topic_label) LIKE ? "
            "OR lower(retrieval_key) LIKE ? "
            "OR lower(summary) LIKE ? "
            "OR lower(reflection) LIKE ? "
            "ORDER BY created_at DESC LIMIT 200",
            (
                f"%{entity_name}%",
                f"%{entity_name}%",
                f"%{entity_name}%",
                f"%{entity_name}%",
            ),
        ).fetchall()
        con.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []

    if len(rows) < 2:
        return []

    reasons: list[str] = []
    qualified_labels: set[str] = set()
    explicit_markers: list[str] = []
    origin_places: set[str] = set()

    escaped = re.escape(base_token)
    marker_re = re.compile(_COLLISION_MARKER_RE.pattern.format(name=escaped),
                           re.IGNORECASE)
    qualified_re = re.compile(
        rf"^{re.escape(base_token)}[-_][a-z0-9][a-z0-9_-]*$",
        re.IGNORECASE,
    )
    origin_re = re.compile(
        rf"\b{re.escape(base_token)}\s+from\s+([A-Z][a-z]+"
        rf"(?:\s+[A-Z][a-z]+)?)\b"
    )

    for summary, reflection, retrieval_key, topic_label, project, tags, _ in rows:
        fields = [summary or "", reflection or "", retrieval_key or "",
                  topic_label or "", project or "", tags or ""]
        text = " ".join(fields)
        topic = (topic_label or "").strip().lower()
        if qualified_re.match(topic) and topic != base_token:
            qualified_labels.add(topic)
        if marker_re.search(text):
            explicit_markers.append(text[:120])
        for match in origin_re.finditer(text):
            origin_places.add(match.group(1).lower())

    if len(qualified_labels) >= 2:
        reasons.append(
            "multiple qualified topic labels for same base name: "
            + ", ".join(sorted(qualified_labels)[:5])
        )
    if explicit_markers:
        reasons.append(
            "explicit same-name disambiguation language observed"
        )
    if len(origin_places) >= 2:
        reasons.append(
            "multiple origin phrases observed for same name: "
            + ", ".join(sorted(origin_places)[:5])
        )

    return reasons


# ---------------------------------------------------------------------------
# Essay tool schemas (for fold driver tool loop, OpenAI function-calling)
# ---------------------------------------------------------------------------

ESSAY_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "essay_get",
        "description": (
            "Read an interpretive essay by entity key. Returns the full "
            "markdown body, title, citations, and cross-references."
        ),
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string",
                    "description": "Entity key (e.g. 'person:kaylee')"}},
            "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "essay_list",
        "description": (
            "List all interpretive essays. Returns entity keys, titles, "
            "patch counts, and last-updated timestamps."
        ),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "essay_edit",
        "description": (
            "Edit an interpretive essay in place — exact string "
            "replacement. If the entity key does not exist, creates a "
            "new essay with new_text as the body. Cite memory references "
            "as [m_<id>] (canonical form). Optionally set citations and "
            "cross_refs as JSON arrays."
        ),
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string",
                    "description": "Entity key"},
            "old_text": {"type": "string",
                         "description": "Exact text to replace"},
            "new_text": {"type": "string",
                         "description": "Replacement text (or full body for new essays)"},
            "title": {"type": "string",
                      "description": "Essay title (optional)"},
            "replace_all": {"type": "boolean",
                            "description": "Replace all occurrences"},
            "citations": {"type": "string",
                          "description": "JSON array of [m_xxxx] IDs cited"},
            "cross_refs": {"type": "string",
                           "description": "JSON array of cross-referenced entity keys"}},
            "required": ["key", "old_text", "new_text"]}}},
]


# ---------------------------------------------------------------------------
# Essay tool execution (for fold driver, operates on the isolated unified DB)
# ---------------------------------------------------------------------------

def exec_essay_tool(
    name: str,
    args: dict,
    db_path: str,
) -> str:
    """Execute an essay tool against the isolated unified DB.

    This mirrors the fold driver's exec_tool() but for essay operations.
    """
    con = sqlite3.connect(db_path)
    _ensure_essays_table(con)

    try:
        if name == "essay_get":
            return _exec_essay_get(con, args)
        elif name == "essay_list":
            return _exec_essay_list(con)
        elif name == "essay_edit":
            return _exec_essay_edit(con, args)
        else:
            return f"Error: unknown essay tool {name}"
    finally:
        con.close()


def _exec_essay_get(con: sqlite3.Connection, args: dict) -> str:
    key = args.get("key", "")
    if not key:
        return "Error: key is required"
    row = con.execute(
        "SELECT entity_key, title, body, citations, cross_refs, "
        "patch_count, created_at, updated_at "
        "FROM essays WHERE entity_key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return f"No essay found for key '{key}'."
    citations = json.loads(row[3])
    cross_refs = json.loads(row[4])
    lines = [
        f"# {row[1] or key}",
        "",
        row[2],
        "",
        "---",
        f"**Entity key:** {row[0]}",
        f"**Patch count:** {row[5]}",
        f"**Updated:** {row[6][:16]}",
    ]
    if citations:
        lines.append(f"**Citations:** {', '.join(citations)}")
    if cross_refs:
        lines.append(f"**Cross-references:** {', '.join(cross_refs)}")
    return "\n".join(lines)


def _exec_essay_list(con: sqlite3.Connection) -> str:
    rows = con.execute(
        "SELECT entity_key, title, patch_count, updated_at "
        "FROM essays ORDER BY entity_key"
    ).fetchall()
    if not rows:
        return "No essays."
    lines = []
    for r in rows:
        title_str = f" — {r[1]}" if r[1] else ""
        lines.append(
            f"**{r[0]}**{title_str} | patches: {r[2]} | updated: {r[3][:16]}"
        )
    return f"{len(rows)} essay{'s' if len(rows) != 1 else ''}:\n\n" + "\n".join(lines)


def _exec_essay_edit(con: sqlite3.Connection, args: dict) -> str:
    key = args.get("key", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    title = args.get("title", "")
    replace_all = args.get("replace_all", False)
    citations_raw = args.get("citations", "")
    cross_refs_raw = args.get("cross_refs", "")

    if not key:
        return "Error: key is required"

    parsed_citations = None
    parsed_cross_refs = None
    if citations_raw:
        if isinstance(citations_raw, list):
            parsed_citations = citations_raw
        else:
            try:
                parsed_citations = json.loads(citations_raw)
                if not isinstance(parsed_citations, list):
                    return "Error: citations must be a JSON array"
            except json.JSONDecodeError as e:
                return f"Error: invalid citations JSON — {e}"
    if cross_refs_raw:
        if isinstance(cross_refs_raw, list):
            parsed_cross_refs = cross_refs_raw
        else:
            try:
                parsed_cross_refs = json.loads(cross_refs_raw)
                if not isinstance(parsed_cross_refs, list):
                    return "Error: cross_refs must be a JSON array"
            except json.JSONDecodeError as e:
                return f"Error: invalid cross_refs JSON — {e}"

    now = datetime.now(timezone.utc).isoformat()
    existing = con.execute(
        "SELECT body, title, citations, cross_refs FROM essays "
        "WHERE entity_key = ?",
        (key,),
    ).fetchone()

    if existing is None:
        create_citations = _inline_memory_citations(new_text)
        # Create new essay
        con.execute(
            "INSERT INTO essays "
            "(entity_key, title, body, citations, cross_refs, "
            "patch_count, created_at, updated_at, curated_version, "
            "verified_hash, verified_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0, '', NULL)",
            (
                key,
                title,
                new_text,
                json.dumps(create_citations),
                json.dumps(parsed_cross_refs or []),
                now,
                now,
            ),
        )
        con.commit()
        return f"Essay '{key}' created ({len(new_text)} chars)."

    # Edit existing: validate first
    if old_text == "":
        return (
            "Error: old_text is empty but essay already exists. "
            "Use a precise old_text replacement for PATCH, or regenerate "
            "with an explicit full-body replacement."
        )

    body = existing[0]
    count = body.count(old_text)
    if count == 0:
        return "Error: old_text not found in essay body."
    if not replace_all and count > 1:
        return (
            f"Error: old_text matches {count} locations — "
            "provide a more specific string or set replace_all=true."
        )

    if replace_all:
        new_body = body.replace(old_text, new_text)
    else:
        new_body = body.replace(old_text, new_text, 1)
    n_replaced = count if replace_all else 1
    body_citations = _inline_memory_citations(new_body)

    new_title = title if title else existing[1]
    full_body_replace = old_text == body
    new_citations = json.dumps(body_citations)

    if parsed_cross_refs is not None:
        if full_body_replace:
            new_cross_refs = json.dumps(parsed_cross_refs)
        else:
            try:
                existing_cross_refs = json.loads(existing[3] or "[]")
            except json.JSONDecodeError:
                existing_cross_refs = []
            new_cross_refs = json.dumps(
                list(dict.fromkeys(existing_cross_refs + parsed_cross_refs))
            )
    else:
        new_cross_refs = existing[3]

    con.execute(
        "UPDATE essays SET body = ?, title = ?, citations = ?, "
        "cross_refs = ?, patch_count = patch_count + 1, updated_at = ?, "
        "verified_hash = '', verified_at = NULL "
        "WHERE entity_key = ?",
        (new_body, new_title, new_citations, new_cross_refs, now, key),
    )
    con.commit()
    return f"Essay updated ({n_replaced} replacement{'s' if n_replaced > 1 else ''})."


# ---------------------------------------------------------------------------
# Fold prompt essay instructions
# ---------------------------------------------------------------------------

ESSAY_FOLD_INSTRUCTIONS = """\

## Interpretive essays (second edit surface)

Alongside the digest, you maintain **per-entity interpretive essays**: deeper
narrative arcs for people, projects, events, or topics that recur across
multiple history windows. Each essay is a self-contained narrative with
[m_<id>] citations linking down to raw memory records.

### Tools
- **essay_list** — see which essays exist (entity keys, titles, patch counts).
- **essay_get(key)** — read an essay's full body.
- **essay_edit(key, old_text, new_text, ...)** — exact-string replacement
  (same semantics as file_edit). Creates the essay if the key doesn't exist.
  Optionally pass citations (JSON array of [m_<id>] strings) and cross_refs
  (JSON array of entity keys).

### When to act

The admission check below tells you which entities from this round's window
qualify for essay work:

{admission_block}

- **PATCH**: the entity already has an essay — update it with the new
  arc-beat from this window. Use essay_edit to make targeted edits.
- **CREATE**: the entity has crossed the recurrence threshold — generate
  a new essay. Use essay_edit with an empty old_text to create.
  Seed the essay from what the digest says about this entity plus the
  raw memory records formed in this and prior windows.
- **SKIP**: below the recurrence threshold — leave in the digest only.

### Rules
1. Essays cite raw memories as [m_<id>] tags (canonical form — NOT [[Mn]]
   handles, which are digest-only transport handles, and NOT [m:<id>] with a
   colon, which does not resolve and does not count as a citation at all).
   A citation attaches to the claim it follows and must be a memory that
   supports THAT claim: a true sentence carrying a citation to some other
   memory reads as verified and is not, and no automated check catches it.
   If a claim rests on several memories, cite them all.
2. Each essay is bounded to {essay_budget} tokens. If an edit pushes past
   this budget, compact the essay the same way you compact the digest.
3. Cross-reference related essays using cross_refs.
4. An essay is a narrative interpretation — not a list of facts. It should
   read as a coherent arc that stands alone.
5. Finish essay work BEFORE emitting the <fold_complete> envelope; the
   driver validates essays as part of the round commit.
"""


def format_essay_instructions(
    admission_actions: list[EssayAction],
    essay_budget: int = 4000,
) -> str:
    """Format the essay-fold instructions for the fold system prompt."""
    if not admission_actions:
        admission_block = "(No entities in this window qualify for essay work.)"
    else:
        lines = []
        for a in admission_actions:
            lines.append(f"- **{a.entity_key}**: {a.action.upper()} "
                         f"(seen in {a.window_count} window{'s' if a.window_count != 1 else ''})")
        admission_block = "\n".join(lines)

    return ESSAY_FOLD_INSTRUCTIONS.format(
        admission_block=admission_block,
        essay_budget=essay_budget,
    )


# ---------------------------------------------------------------------------
# Phase 3 — Meta-review: scan digest for entities, compare against essays
# ---------------------------------------------------------------------------

# Patterns for extracting entity references from a standing digest.
# People: **Name** appearing in a narrative context (bold proper nouns)
_PERSON_BOLD_RE = re.compile(r"\*\*([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\*\*")
# Inline capitalized proper nouns, for digests that don't bold names
_PERSON_INLINE_1W_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
_PERSON_INLINE_2W_RE = re.compile(r"\b([A-Z][a-z]{2,}\s[A-Z][a-z]{2,})\b")
_INLINE_MIN_FREQ = 3
# Projects: bracketed project refs or bold project-like names
_PROJECT_RE = re.compile(r"\bproject[:\-](\S+)")
# Memory refs embedded in digest text
_DIGEST_MEM_REF_RE = re.compile(r"\[m_([0-9a-fA-F]{6,40})\]")

_NOISE_WORDS: frozenset[str] = frozenset({
    "The", "This", "That", "Note", "See", "Also", "But", "And", "For",
    "Not", "When", "Where", "What", "How", "Why", "Who", "Which",
    "All", "Any", "Each", "Every", "Some", "Most", "Many", "Few",
    "None", "Both", "Other", "Same", "Such", "More", "Less", "Very",
    "Still", "Just", "Even", "Only", "Already", "Then", "Here", "There",
    "Now", "Yet", "Once", "Never", "Always", "About", "After", "Before",
    "Between", "During", "Since", "Until", "With", "Without", "Within",
    "Into", "From", "Over", "Under", "True", "False", "Yes", "Done",
    "New", "Old", "Good", "Bad", "Key", "Set", "Get", "Put", "Run",
    "Use", "Try", "Let", "May", "Got", "Did", "Has", "Had", "Was",
    "Were", "Been", "Being",
    "She", "Her", "His", "Him", "They", "Them", "Their",
    "Timeline", "Summary", "Current", "State", "Open", "Issues",
    "Goals", "Glossary", "Phase", "Round", "Step", "Section",
    "Pipeline", "Memory", "Router", "Worker", "Agent", "System",
    "Entry", "Table", "Index", "Query", "Score", "Status",
    "Decision", "Action", "Result", "Error", "Warning", "Info",
    "First", "Last", "Next", "Previous", "Final", "Initial",
    "Daily", "Weekly", "Monthly", "Nightly",
    "Fold", "Digest", "Essay", "Review", "Report", "Draft",
    "Code", "Paper", "Grant", "Research", "Active", "Tier",
    "Small", "Three", "Two", "While", "Multi", "Therapy",
    "Location", "Childcare", "Summer",
    "June", "July", "August", "September", "October", "November",
    "December", "January", "February", "March", "April",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    "Father", "Mother", "Mommy", "Daddy", "Master", "Administrator",
    "Silver", "Veldt", "Jail", "Kid", "Liar",
    "Agentic", "Amazon", "Ansible", "Anthropic", "Architecture",
    "Backend", "Benchmark", "Bridge", "Carderock",
    "Claude", "Coco", "Codex", "Compiler", "Complete",
    "Config", "Context", "Controller", "Declaration",
    "Day", "Disk", "Docker", "Edit", "Email", "Eval", "Exa", "Execute",
    "Faculty", "Fix", "Foreach", "Frontend",
    "Fulltext", "Glasswing", "Gmail", "Growth", "Guni", "Health", "Hypatia",
    "Infrastructure", "Interactive", "Intermediate", "Investment",
    "Journal", "Leiden", "Lit", "Literature", "Mesh",
    "Mythos", "News", "Payment", "Per", "Personal", "Personality",
    "Plan", "Pre", "Pro", "Proposal",
    "Qwen", "Recursive", "Reme", "Researcher", "Roth",
    "Seed", "Selection", "Self", "Session", "Sol", "Standalone",
    "Strategy", "Submodular", "Survey", "Tanglewood", "Test", "Thu",
    "Vanguard", "Verifier", "Wake", "Award", "Web", "Website", "Webhook",
})

_POSSESSIVE_RE = re.compile(r"\b([A-Z][a-z]{2,})'s\b")
_DICT_WORDS: frozenset[str] | None = None


def _load_dict_words() -> frozenset[str]:
    global _DICT_WORDS
    if _DICT_WORDS is not None:
        return _DICT_WORDS
    try:
        with open("/usr/share/dict/words") as f:
            _DICT_WORDS = frozenset(
                w.strip().lower() for w in f if 3 <= len(w.strip()) <= 20
            )
    except (FileNotFoundError, OSError):
        _DICT_WORDS = frozenset()
    return _DICT_WORDS


class MetaReviewItem:
    """One finding from a meta-review scan."""
    __slots__ = ("entity_key", "action", "reason", "detail")

    SEED = "seed"
    LUMPY = "lumpy"
    STALE = "stale"
    DIGEST_TIGHTEN = "digest_tighten"
    ENTITY_COLLISION = "entity_collision"
    SKILL_REQUIRES_ATTENTION = "skill_requires_attention"
    SKILL_FOR_REVIEW = "skill_for_review"
    SKILL_CANDIDATE = "skill_candidate"

    def __init__(self, entity_key: str, action: str, reason: str,
                 detail: str = ""):
        self.entity_key = entity_key
        self.action = action
        self.reason = reason
        self.detail = detail

    def __repr__(self) -> str:
        return (f"MetaReviewItem({self.entity_key!r}, {self.action!r}, "
                f"{self.reason!r})")


_DB_CORROBORATE_MIN_FREQ = 3


def _db_corroborate_names(db_path: str) -> set[str]:
    """Query topic_labels in a memory DB for capitalized names (possessive or
    frequent) to rescue dict-word personal names the digest-only heuristic
    rejects.  Returns a set of Title-cased names corroborated by the DB."""
    import os
    if not os.path.exists(db_path):
        return set()
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT topic_label FROM memories WHERE topic_label != ''"
        ).fetchall()
        con.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return set()
    poss_re = re.compile(r"\b([A-Z][a-z]{2,})'s\b")
    cap_re = re.compile(r"\b([A-Z][a-z]{2,})\b")
    names: set[str] = set()
    freq: dict[str, int] = {}
    for (tl,) in rows:
        for m in poss_re.finditer(tl):
            n = m.group(1)
            if n not in _NOISE_WORDS:
                names.add(n)
        for m in cap_re.finditer(tl):
            n = m.group(1)
            if n not in _NOISE_WORDS:
                freq[n] = freq.get(n, 0) + 1
    for n, c in freq.items():
        if c >= _DB_CORROBORATE_MIN_FREQ:
            names.add(n)
    return names


def _canonicalize_aliases(
    entities: dict[str, list[str]],
    digest_text: str | None = None,
) -> dict[str, list[str]]:
    """Fold first-name and surname standalone keys into canonical full-name
    keys when a two-word entity exists.

    For each ``person:first-last`` key, standalone ``person:first`` and
    ``person:last`` are merged (refs union'd) and removed.  This prevents
    the same person appearing under 2–3 separate keys.

    When *digest_text* is provided, standalone pairs whose full name
    ("First Last") appears at least once in the digest are also merged
    into a new two-word key even if the two-word entity was not
    independently extracted.
    """
    person_keys = [k for k in entities if k.startswith("person:")]
    two_word = {}
    for k in person_keys:
        parts = k.split(":", 1)[1].split("-")
        if len(parts) == 2:
            two_word[k] = (parts[0], parts[1])

    remove: set[str] = set()
    for full_key, (first, last) in two_word.items():
        first_key = f"person:{first}"
        last_key = f"person:{last}"
        for partial_key in (first_key, last_key):
            if partial_key in entities and partial_key != full_key:
                for ref in entities[partial_key]:
                    if ref not in entities[full_key]:
                        entities[full_key].append(ref)
                remove.add(partial_key)

    if digest_text:
        single_word = [
            k for k in entities
            if k.startswith("person:")
            and "-" not in k.split(":", 1)[1]
            and k not in remove
        ]
        claimed: set[str] = set()
        for i, k1 in enumerate(single_word):
            if k1 in claimed:
                continue
            name1 = k1.split(":", 1)[1]
            for k2 in single_word[i + 1:]:
                if k2 in claimed:
                    continue
                name2 = k2.split(":", 1)[1]
                for first, last, fk, lk in [
                    (name1, name2, k1, k2),
                    (name2, name1, k2, k1),
                ]:
                    full_name = f"{first.title()} {last.title()}"
                    if full_name in digest_text:
                        full_key = f"person:{first}-{last}"
                        if full_key not in entities:
                            entities[full_key] = []
                        for ref in entities.get(fk, []):
                            if ref not in entities[full_key]:
                                entities[full_key].append(ref)
                        for ref in entities.get(lk, []):
                            if ref not in entities[full_key]:
                                entities[full_key].append(ref)
                        remove.add(fk)
                        remove.add(lk)
                        claimed.add(k1)
                        claimed.add(k2)
                        break
                if k1 in claimed:
                    break

    for k in remove:
        if k in entities:
            del entities[k]

    return entities


def extract_digest_entities(
    digest_text: str,
    db_path: str | None = None,
) -> dict[str, list[str]]:
    """Extract entity candidates from digest text.

    Uses three strategies for person detection:
    1. Bold names (**Name**) — always included regardless of frequency
    2. Two-word inline names (First Last) — frequency threshold, no dict filter
    3. Single-word inline names — frequency threshold, filtered by system
       dictionary to exclude common English words (dictionary words accepted
       if they appear in possessive form in the digest or are corroborated
       by frequency/possessives in the memory DB topic_labels)

    When *db_path* is provided, topic_labels are scanned for additional
    person signals: names with possessives or appearing 3+ times in
    topic_labels are accepted even if they are dictionary words.

    Returns a dict mapping entity keys to the list of memory ref IDs
    found near them in the digest (may be empty).
    """
    entities: dict[str, list[str]] = {}

    for m in _PERSON_BOLD_RE.finditer(digest_text):
        name = m.group(1).strip()
        if len(name) < 2 or name in _NOISE_WORDS:
            continue
        if any(w in _NOISE_WORDS for w in name.split()):
            continue
        key = f"person:{name.lower().replace(' ', '-')}"
        if key not in entities:
            entities[key] = []

    possessive_names: set[str] = set()
    for m in _POSSESSIVE_RE.finditer(digest_text):
        name = m.group(1)
        if name not in _NOISE_WORDS:
            possessive_names.add(name)

    db_corroborated: set[str] = set()
    if db_path:
        db_corroborated = _db_corroborate_names(db_path)

    accepted_names = possessive_names | db_corroborated

    inline_2w_counts: dict[str, int] = {}
    for m in _PERSON_INLINE_2W_RE.finditer(digest_text):
        name = m.group(1).strip()
        if any(w in _NOISE_WORDS for w in name.split()):
            continue
        inline_2w_counts[name] = inline_2w_counts.get(name, 0) + 1

    for name, count in inline_2w_counts.items():
        if count >= _INLINE_MIN_FREQ:
            key = f"person:{name.lower().replace(' ', '-')}"
            if key not in entities:
                entities[key] = []

    dict_words = _load_dict_words()

    inline_1w_counts: dict[str, int] = {}
    for m in _PERSON_INLINE_1W_RE.finditer(digest_text):
        name = m.group(1).strip()
        if name in _NOISE_WORDS:
            continue
        inline_1w_counts[name] = inline_1w_counts.get(name, 0) + 1

    for name, count in inline_1w_counts.items():
        if count < _INLINE_MIN_FREQ:
            continue
        if dict_words and name.lower() in dict_words:
            if name not in accepted_names:
                continue
        key = f"person:{name.lower().replace(' ', '-')}"
        if key not in entities:
            entities[key] = []

    for m in _PROJECT_RE.finditer(digest_text):
        proj = m.group(1).strip().rstrip(".,;:)*")
        if proj:
            key = f"project:{proj}"
            if key not in entities:
                entities[key] = []

    for m in _DIGEST_MEM_REF_RE.finditer(digest_text):
        mid = m.group(1)
        start = max(0, m.start() - 300)
        ctx = digest_text[start:m.start()]
        for ek in entities:
            name_part = ek.split(":", 1)[1].replace("-", " ")
            if name_part.lower() in ctx.lower():
                entities[ek].append(mid)

    entities = _canonicalize_aliases(entities, digest_text=digest_text)

    return entities


def scan_meta_review(
    db_path: str,
    digest_text: str,
    lumpy_threshold: int = 10,
    skills_owner: str | None = None,
    skills_root: str | None = None,
    procedural_window_text: str = "",
) -> list[MetaReviewItem]:
    """Run the meta-review scan: compare digest entities against the essays table.

    Returns a list of MetaReviewItems describing recommended actions.
    Does NOT execute any changes — this is the report-only pass.

    Roles per decision 31:
      1. SEEDING — entities in digest that have no essay → recommend creation
      2. HYGIENE — essays with high patch counts (lumpy), entities in essays
         but not in digest (stale)
      3. DIGEST CONSISTENCY — entities that have well-maintained essays and
         could be tightened to pointers in the digest
    """
    items: list[MetaReviewItem] = []
    digest_entities = extract_digest_entities(digest_text, db_path=db_path)

    con = sqlite3.connect(db_path)
    _ensure_essays_table(con)

    existing_essays: dict[str, tuple[str, int, str]] = {}
    for row in con.execute(
        "SELECT entity_key, title, patch_count, body FROM essays"
    ).fetchall():
        existing_essays[row[0]] = (row[1], row[2], row[3])

    # Entity hygiene: warn when one key appears to collect multiple
    # same-name entities.  The generator can still proceed, but the audit
    # should tell us to split keys before the essay hardens the conflation.
    for ek in sorted(set(digest_entities) | set(existing_essays)):
        collision_reasons = _detect_entity_collisions(db_path, ek)
        if collision_reasons:
            entity_name = _entity_name_from_key(ek)
            prefix = ek.split(":", 1)[0] if ":" in ek else "entity"
            base = entity_name.split()[0].lower()
            items.append(MetaReviewItem(
                ek, MetaReviewItem.ENTITY_COLLISION,
                "possible entity collision — memories may describe multiple "
                "distinct entities",
                detail=(
                    f"{'; '.join(collision_reasons[:3])}. "
                    f"Consider disambiguating with qualified keys such as "
                    f"{prefix}:{base}-context."
                ),
            ))

    # Role 1: SEEDING — digest entities without essays
    for ek, mem_refs in digest_entities.items():
        if ek not in existing_essays:
            if not mem_refs:
                continue
            items.append(MetaReviewItem(
                ek, MetaReviewItem.SEED,
                f"entity appears in digest but has no essay",
                detail=f"linked memory refs: {mem_refs[:5]}",
            ))

    # Role 2: HYGIENE
    # Lumpy essays (many patches without regeneration)
    for ek, (title, patch_count, body) in existing_essays.items():
        if patch_count >= lumpy_threshold:
            items.append(MetaReviewItem(
                ek, MetaReviewItem.LUMPY,
                f"essay has {patch_count} patches — consider regeneration",
                detail=f"title: {title}",
            ))

    # Stale essays (entity not referenced in digest)
    for ek in existing_essays:
        if ek not in digest_entities:
            name_part = ek.split(":", 1)[1].replace("-", " ")
            if name_part.lower() not in digest_text.lower():
                items.append(MetaReviewItem(
                    ek, MetaReviewItem.STALE,
                    "essay entity not found in current digest",
                    detail=f"title: {existing_essays[ek][0]}",
                ))

    # Role 3: DIGEST CONSISTENCY — well-maintained essays that could let
    # the digest entry be tightened to a pointer
    for ek in digest_entities:
        if ek in existing_essays:
            title, patch_count, body = existing_essays[ek]
            if body and len(body) > 200:
                items.append(MetaReviewItem(
                    ek, MetaReviewItem.DIGEST_TIGHTEN,
                    "essay exists and is substantial — digest entry could be "
                    "tightened to a pointer",
                    detail=f"essay: {len(body)} chars, {patch_count} patches",
                ))

    con.close()

    # Governed procedural memory shares the report-only meta-review surface
    # while retaining a separate filesystem store.  The fold driver supplies
    # the agent owner and the just-completed window; legacy callers that omit
    # them retain essay-only behavior.
    if skills_owner:
        try:
            store = SkillStore(skills_owner, root=skills_root)
            for finding in store.scan_meta_review():
                action = (
                    MetaReviewItem.SKILL_REQUIRES_ATTENTION
                    if finding.tier == 1
                    else MetaReviewItem.SKILL_FOR_REVIEW
                )
                items.append(MetaReviewItem(
                    f"skill:{finding.card_id}",
                    action,
                    finding.reason,
                    detail=finding.detail,
                ))
        except (OSError, SkillCardError) as exc:
            items.append(MetaReviewItem(
                f"skill:{skills_owner}",
                MetaReviewItem.SKILL_REQUIRES_ATTENTION,
                "skill-card meta-review could not read the governed store",
                detail=str(exc),
            ))

        signal = detect_formation_signal(procedural_window_text)
        if signal.eligible:
            items.append(MetaReviewItem(
                f"skill-candidate:{skills_owner}",
                MetaReviewItem.SKILL_CANDIDATE,
                "completed procedure passes the structural formation gate",
                detail=(
                    f"{signal.rationale}; observed details: "
                    f"{list(signal.details)[:8]}"
                ),
            ))
    return items


def format_meta_review_report(items: list[MetaReviewItem]) -> str:
    """Format a human-readable meta-review report."""
    if not items:
        return "Meta-review: no findings. Digest and essay table are consistent."

    by_action: dict[str, list[MetaReviewItem]] = {}
    for item in items:
        by_action.setdefault(item.action, []).append(item)

    sections = []
    section_order = [
        (MetaReviewItem.ENTITY_COLLISION,
         "Warnings — possible entity collisions"),
        (MetaReviewItem.SEED, "Seeding — entities needing essays"),
        (MetaReviewItem.LUMPY, "Hygiene — lumpy essays (high patch count)"),
        (MetaReviewItem.STALE, "Hygiene — stale essays (not in digest)"),
        (MetaReviewItem.DIGEST_TIGHTEN,
         "Digest consistency — candidates for pointer tightening"),
        (MetaReviewItem.SKILL_REQUIRES_ATTENTION,
         "Skill cards, Tier 1 — Requires attention"),
        (MetaReviewItem.SKILL_FOR_REVIEW,
         "Skill cards, Tier 2 — For review"),
        (MetaReviewItem.SKILL_CANDIDATE,
         "Skill cards — formation candidates (report only)"),
    ]
    for action, header in section_order:
        group = by_action.get(action, [])
        if not group:
            continue
        lines = [f"### {header} ({len(group)})"]
        for item in sorted(group, key=lambda x: x.entity_key):
            lines.append(f"- **{item.entity_key}**: {item.reason}")
            if item.detail:
                lines.append(f"  {item.detail}")
        sections.append("\n".join(lines))

    header = f"# Meta-Review Report\n\n{len(items)} finding(s)\n"
    return header + "\n\n".join(sections)


def build_seeding_prompt(
    entity_key: str,
    digest_text: str,
    memory_summaries: list[tuple[str, str]],
    essay_budget: int = 4000,
) -> str:
    """Build the LLM prompt for seeding a new essay from digest + raw memories.

    .. deprecated:: Use :func:`generate_essay` instead — this function
       pre-dates the ReAct generator and does not support memory retrieval.

    memory_summaries: list of (memory_id, summary_text) tuples.
    """
    mem_block = ""
    if memory_summaries:
        lines = []
        for mid, summ in memory_summaries:
            lines.append(f"- [m_{mid}] {summ[:300]}")
        mem_block = (
            f"### Raw memories linked to {entity_key} "
            f"({len(memory_summaries)} records)\n"
            + "\n".join(lines) + "\n"
        )
    else:
        mem_block = f"(No raw memories found for {entity_key}.)\n"

    return (
        f"### META-REVIEW SEEDING\n"
        f"Create a new interpretive essay for **{entity_key}**.\n\n"
        f"### Current digest (read-only context)\n"
        f"{digest_text[:8000]}\n\n"
        f"{mem_block}\n"
        f"Use essay_edit to create the essay for {entity_key}. "
        f"Write a coherent narrative arc — not a list of facts. "
        f"Cite raw memories as [m_<id>] tags. "
        f"Keep within {essay_budget} tokens. "
        f"Cross-reference related entities via cross_refs.\n\n"
        f"When done, emit:\n"
        f"<fold_complete><covering_from>meta-review</covering_from>"
        f"<covering_to>{entity_key}</covering_to></fold_complete>"
    )


# ---------------------------------------------------------------------------
# ReAct essay generator (replaces build_seeding_prompt + inline fold essays)
# ---------------------------------------------------------------------------

GENERATOR_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "memory_search",
        "description": (
            "Full-text search over raw memory records. Returns matching "
            "memory IDs with summaries, topics, and projects. Use this to "
            "find memories relevant to the entity you are writing about. "
            "Try multiple queries (entity name, related topics, related people)."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "Search keywords (name, topic, phrase)"},
            "limit": {"type": "integer",
                      "description": "Max results (default 20)"},
            "created_after": {
                "type": "string",
                "description": (
                    "Optional ISO timestamp; restrict results to memories "
                    "created after this time."
                ),
            },
            "entity_name": {
                "type": "string",
                "description": (
                    "Optional entity name; restrict results to memories whose "
                    "topic, project, or retrieval key mentions this entity."
                ),
            }},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "memory_get",
        "description": (
            "Retrieve the full record for one memory by ID. Returns "
            "summary, reflection, trace, topic, project, tags, and outcome. "
            "Use after memory_search to read details of memories you intend "
            "to cite."
        ),
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string",
                   "description": "Memory ID (hex string, e.g. 'abc123def456')"}},
            "required": ["id"]}}},
    {"type": "function", "function": {
        "name": "essay_edit",
        "description": (
            "Write or update the essay. For new essays pass old_text='' and "
            "new_text=<full body>. For edits pass the exact text to replace. "
            "Cite memories as [m_<id>]. Optionally set title, citations "
            "(JSON array of m_<id> strings), and cross_refs (JSON array of "
            "entity keys)."
        ),
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "Entity key"},
            "old_text": {"type": "string",
                         "description": "Exact text to replace (empty for new)"},
            "new_text": {"type": "string",
                         "description": "Replacement text (or full body)"},
            "title": {"type": "string", "description": "Essay title"},
            "replace_all": {"type": "boolean",
                            "description": "Replace all occurrences"},
            "citations": {"type": "string",
                          "description": "JSON array of cited [m_<id>] IDs"},
            "cross_refs": {"type": "string",
                           "description": "JSON array of related entity keys"}},
            "required": ["key", "old_text", "new_text"]}}},
]


def _memory_search_filters(
    created_after: str | None = None,
    entity_name: str | None = None,
) -> tuple[str, list[str]]:
    filters = []
    params: list[str] = []
    if created_after:
        filters.append("m.created_at > ?")
        params.append(created_after)
    if entity_name:
        pattern = f"%{entity_name}%"
        filters.append(
            "(m.topic_label LIKE ? OR m.project LIKE ? OR m.retrieval_key LIKE ?)"
        )
        params.extend([pattern, pattern, pattern])
    if not filters:
        return "", []
    return " AND " + " AND ".join(filters), params


def _exec_memory_search(
    db_path: str,
    query: str,
    limit: int = 20,
    created_after: str | None = None,
    entity_name: str | None = None,
) -> str:
    """FTS5 search over the unified DB's memories, LIKE fallback."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = []
    filter_sql, filter_params = _memory_search_filters(
        created_after=created_after, entity_name=entity_name,
    )
    if not query:
        rows = con.execute(
            "SELECT m.id, m.summary, m.topic_label, m.project FROM memories m "
            f"WHERE 1 = 1 {filter_sql} "
            "ORDER BY created_at DESC LIMIT ?",
            [*filter_params, limit],
        ).fetchall()
    else:
        try:
            rows = con.execute(
                "SELECT f.id, m.summary, m.topic_label, m.project "
                "FROM memories_fts f "
                "JOIN memories m ON m.id = f.id "
                "WHERE memories_fts MATCH ? "
                f"{filter_sql} "
                "ORDER BY -bm25(memories_fts) "
                "LIMIT ?",
                [query, *filter_params, limit],
            ).fetchall()
        except sqlite3.OperationalError:
            pattern = f"%{query}%"
            rows = con.execute(
                "SELECT m.id, m.summary, m.topic_label, m.project FROM memories m "
                "WHERE (m.summary LIKE ? OR m.reflection LIKE ? "
                "OR m.retrieval_key LIKE ? OR m.topic_label LIKE ?) "
                f"{filter_sql} "
                "ORDER BY created_at DESC LIMIT ?",
                [pattern, pattern, pattern, pattern, *filter_params, limit],
            ).fetchall()
    con.close()
    if not rows:
        return f"No memories found for query '{query}'."
    lines = []
    for mid, summ, topic, proj in rows:
        meta = []
        if topic:
            meta.append(f"topic={topic}")
        if proj:
            meta.append(f"project={proj}")
        meta_str = f" ({', '.join(meta)})" if meta else ""
        lines.append(f"- [m_{mid}]{meta_str}: {summ[:200]}")
    return f"{len(rows)} result(s):\n" + "\n".join(lines)


def _exec_memory_get(db_path: str, memory_id: str) -> str:
    """Fetch one full memory record by ID."""
    clean_id = memory_id
    if clean_id.startswith("m_"):
        clean_id = clean_id[2:]
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = con.execute(
        "SELECT id, created_at, summary, reflection, trace, "
        "retrieval_key, topic_label, project, tags, outcome "
        "FROM memories WHERE id = ?",
        (clean_id,),
    ).fetchone()
    con.close()
    if row is None:
        return f"No memory found with ID '{memory_id}'."
    mid, created, summ, refl, trace, rkey, topic, proj, tags, outcome = row
    lines = [
        f"**Memory [m_{mid}]**",
        f"- Created: {created[:16]}",
        f"- Summary: {summ}",
    ]
    if refl:
        lines.append(f"- Reflection: {refl}")
    if topic:
        lines.append(f"- Topic: {topic}")
    if proj:
        lines.append(f"- Project: {proj}")
    if tags:
        lines.append(f"- Tags: {tags}")
    if outcome:
        lines.append(f"- Outcome: {outcome}")
    if trace:
        lines.append(f"- Trace: {trace[:500]}")
    return "\n".join(lines)


def exec_generator_tool(name: str, args: dict, db_path: str) -> str:
    """Execute a generator tool (memory_search, memory_get, essay_edit)."""
    if name == "memory_search":
        query = args.get("query", "")
        if not query:
            return "Error: query is required"
        limit = args.get("limit", 20)
        return _exec_memory_search(
            db_path, query, int(limit),
            created_after=args.get("created_after") or None,
            entity_name=args.get("entity_name") or None,
        )
    elif name == "memory_get":
        mid = args.get("id", "")
        if not mid:
            return "Error: id is required"
        return _exec_memory_get(db_path, mid)
    elif name == "essay_edit":
        return exec_essay_tool(name, args, db_path)
    else:
        return f"Error: unknown tool '{name}'"


# ---------------------------------------------------------------------------
# Essay type detection and section schemas (locked 2026-07-13)
# ---------------------------------------------------------------------------

def _essay_type_from_key(entity_key: str) -> str:
    """Determine the essay type from the entity key prefix."""
    if entity_key.startswith("project:"):
        return "project"
    if entity_key.startswith("event:"):
        return "event"
    return "person"


_PERSON_SECTION_SCHEMA = """\
### Required sections (in this order)

1. **Identity** — One sentence: who they are and the relationship type \
(collaborator, family, colleague, mentee, etc.). This grounds everything below.
2. **Timeline** — Chronological. Every entry is a dated fact with at least one \
[m_<id>] citation. NO interpretation here — facts only.
3. **Narrative** — The most important events and patterns, with interpretation. \
Every interpretive claim must still anchor to at least one [m_<id>] citation.
4. **Reflection** — Where the relationship currently stands. Synthesis is \
allowed, but trace claims to memories where possible.
5. **Open threads** — Unresolved commitments, pending follow-ups, and \
outstanding questions. Cite the memory that created each thread where possible.

### Facts-vs-interpretation boundary (HARD RULE)
- **Timeline** = cited fact ONLY. No adjectives, no judgment, no interpretation.
- **Narrative** = interprets, but every claim anchors to [m_<id>] citations.
- **Reflection** = may synthesize, but should trace to memories where possible.
- This separation is what makes the essay auditable rather than confabulated.

### Cross-references
- Always populate the `cross_refs` JSON array with related entity keys.
- Weave references to related people/projects into Timeline and Narrative \
where they carry meaning in context.
- Do NOT create a separate "Related" section — for people, relationships are \
story, not an index."""

_PROJECT_SECTION_SCHEMA = """\
### Required sections (in this order)

1. **Goal** — One sentence: what the project aims to achieve and why it exists.
2. **Decision arc** — Key decisions in chronological order, each cited with \
[m_<id>]. What was decided, when, and why.
3. **Current state** — Where the project stands now. Active work, blocking \
issues, recent milestones.
4. **Open issues** — Unresolved problems, pending decisions, and next steps.
5. **Related** — People involved and their roles, dependencies on other \
projects, what this project blocks or enables. Enumerate these connections \
explicitly — projects are hubs and readers need connections listed, not hidden \
in prose.

### Cross-references
- Always populate the `cross_refs` JSON array with related entity keys.
- The Related section provides human-readable enumeration of these connections."""

_EVENT_SECTION_SCHEMA = """\
### Required sections (in this order)

1. **What happened** — Factual account of the event, cited with [m_<id>].
2. **Significance** — Why the event matters. Interpretation and context, \
anchored to citations.
3. **Aftermath** — Consequences, follow-up actions, and current state.

### Cross-references
- Always populate the `cross_refs` JSON array with related entity keys.
- Weave references to related people/projects into the narrative where they \
carry meaning — no separate Related section for events."""


def _section_schema_for_type(entity_type: str) -> str:
    """Return the section schema string for the given entity type."""
    if entity_type == "project":
        return _PROJECT_SECTION_SCHEMA
    if entity_type == "event":
        return _EVENT_SECTION_SCHEMA
    return _PERSON_SECTION_SCHEMA


ESSAY_GENERATOR_SYSTEM = """\
You are the essay generator for {agent}'s memory system.

Your task: produce one interpretive dossier for a single {entity_type} entity.
The essay is a self-contained document — grounded in cited memory records —
that captures everything the agent knows about this entity in a structured,
auditable form.

## Section structure

{section_schema}

## Process

1. **RETRIEVE** — Use memory_search to find all memories relevant to the
   entity. Try multiple queries: the entity name, related topics, related
   people, specific events mentioned in the digest. Use memory_get to read
   full details of the most relevant results.

2. **SYNTHESIZE** — Identify the narrative arc. What is the story of this
   entity as seen through the agent's memories? What themes recur? What has
   changed over time? What is the entity's significance?

3. **WRITE** — Use essay_edit to create or update the essay:
   - Follow the section structure above exactly (same order, same headings)
   - Cite every factual claim with at least one [m_<id>] memory reference
   - Stay within {budget} tokens
   - Set a descriptive title via the title parameter
   - Include a citations array (JSON) listing all [m_<id>] IDs cited
   - Include cross_refs (JSON array) for related entity keys

4. **COMPLETE** — When the essay is written and validated, emit:
   <fold_complete><covering_from>generate</covering_from>\
<covering_to>{entity_key}</covering_to></fold_complete>

## Rules
- Cite memories as [m_<id>] — never use [[...]] placeholder tokens
- Every factual claim must be grounded in at least one cited memory
- The essay should stand alone — a reader with no other context should
  understand the entity's significance
- Do NOT edit any other essay or the digest — only the essay for {entity_key}
"""

ESSAY_GENERATOR_USER_CREATE = """\
## Entity
**{entity_key}** (type: {entity_type}) — new essay (no existing essay)

## Current Digest (read-only context)
{digest_excerpt}

## Instructions
Create a new interpretive essay for **{entity_key}**. Begin by searching for
all memories related to **{entity_name}**. Try multiple queries to ensure
thorough coverage, then write the essay following the section structure
described in the system prompt.

If the retrieved memories appear to describe fundamentally different entities
(for example, different people with the same name, or different projects
conflated under one key), do not force them into a single clean narrative.
Flag it in the Open threads / Open issues section as: "Possible entity collision:
memories may describe multiple distinct people/projects. Consider disambiguation."
"""

ESSAY_GENERATOR_USER_PATCH = """\
## Entity
**{entity_key}** (type: {entity_type}) — update existing essay

## Current Digest (read-only context)
{digest_excerpt}

## Existing Essay
Title: {existing_title}

{existing_body}

## New Material to Review

The following memories were created or updated after the essay's last
modification ({essay_updated_at}). Review each one. If any contain
significant new information (factual changes, new events, new decisions),
weave them into the essay using essay_edit. If none are significant, state
that explicitly before completing.

{prefetched_memories}

## Instructions
The existing essay for **{entity_key}** needs updating with new information.
Search for memories about **{entity_name}** created or updated after
**{essay_updated_at}**. The digest may not mention them — that's expected.
Pull and read anything you find before deciding whether it is significant
enough to weave into the existing essay. Use essay_edit to make targeted
updates that preserve the section structure from the system prompt.
Add new citations for any new material integrated.
"""

ESSAY_GENERATOR_USER_REGENERATE = """\
## Entity
**{entity_key}** (type: {entity_type}) — full regeneration (rewrite from scratch)

## Current Digest (read-only context)
{digest_excerpt}

## Previous Essay (to be replaced)
Title: {existing_title}

{existing_body}

## Instructions
Regenerate the essay for **{entity_key}** from scratch. The previous essay
accumulated too many patches and needs a clean rewrite. Search for all
relevant memories and write a fresh essay following the section structure
from the system prompt. You may reference the previous essay for context
but write independently.
"""

_GENERATE_COMPLETE_RE = re.compile(
    r"<fold_complete>.*?<covering_to>(.*?)</covering_to>.*?</fold_complete>",
    re.DOTALL,
)


def generate_essay(
    entity_key: str,
    digest_text: str,
    db_path: str,
    llm_fn,
    essay_budget: int = 4000,
    action: str = "create",
    existing_body: str = "",
    existing_title: str = "",
    max_rounds: int = 12,
    known_ids: set[str] | None = None,
    ntokens_fn=None,
    log_fn=None,
    agent: str = "agent",
) -> tuple[bool, str]:
    """Generate or update one essay via a bounded ReAct loop.

    This is the single entry point for all essay writes (CREATE, PATCH,
    REGENERATE).  The LLM receives the digest as read-only context, plus
    memory_search and memory_get tools for retrieving relevant memories
    from the unified DB, then writes the essay via essay_edit.

    Args:
        entity_key: Target entity (e.g. ``person:kaylee``)
        digest_text: Current standing digest (read-only context)
        db_path: Path to the unified DB (memories + essays tables)
        llm_fn: ``(messages, tools) -> (content, tool_calls, usage)``
            where messages/tools are OpenAI format and tool_calls are
            raw dicts ``[{"id":..., "function":{"name":..., "arguments":...}}]``
        essay_budget: Max token budget per essay
        action: One of ``"create"``, ``"patch"``, ``"regenerate"``
        existing_body: Current essay body (for patch/regenerate)
        existing_title: Current essay title (for patch/regenerate)
        max_rounds: Maximum ReAct loop iterations
        known_ids: Set of valid memory IDs for citation validation
        ntokens_fn: Token counting function (or None to skip budget check)
        log_fn: Logging callback ``(msg: str) -> None``
        agent: Agent name for prompt personalization

    Returns:
        ``(success, message)`` tuple
    """
    if log_fn is None:
        log_fn = lambda msg: None

    if known_ids is None:
        con = sqlite3.connect(db_path)
        rows = con.execute("SELECT id FROM memories").fetchall()
        con.close()
        known_ids = {r[0] for r in rows}

    entity_name = (
        entity_key.split(":", 1)[1].replace("-", " ")
        if ":" in entity_key else entity_key
    )

    entity_type = _essay_type_from_key(entity_key)
    section_schema = _section_schema_for_type(entity_type)

    system_prompt = ESSAY_GENERATOR_SYSTEM.format(
        agent=agent, budget=essay_budget, entity_key=entity_key,
        entity_type=entity_type, section_schema=section_schema,
    )

    fmt = dict(
        entity_key=entity_key,
        entity_name=entity_name,
        entity_type=entity_type,
        digest_excerpt=digest_text[:8000],
        existing_body=existing_body,
        existing_title=existing_title,
        essay_updated_at="unknown",
        prefetched_memories="No new memories found since the essay was last updated.",
    )
    if action == "create":
        con = sqlite3.connect(db_path)
        _ensure_essays_table(con)
        row = con.execute(
            "SELECT title, body, updated_at FROM essays WHERE entity_key = ?",
            (entity_key,),
        ).fetchone()
        con.close()
        if row:
            action = "patch"
            existing_title = row[0]
            existing_body = row[1]
            essay_updated_at = row[2] or "unknown"
            fmt["existing_title"] = existing_title
            fmt["existing_body"] = existing_body
            fmt["essay_updated_at"] = essay_updated_at

    if action == "patch":
        con = sqlite3.connect(db_path)
        _ensure_essays_table(con)
        row = con.execute(
            "SELECT title, body, updated_at FROM essays WHERE entity_key = ?",
            (entity_key,),
        ).fetchone()
        con.close()
        if row:
            existing_title = row[0]
            existing_body = row[1]
            essay_updated_at = row[2] or "unknown"
            fmt["existing_title"] = existing_title
            fmt["existing_body"] = existing_body
            fmt["essay_updated_at"] = essay_updated_at

    if action == "patch":
        essay_updated_at = fmt["essay_updated_at"]
        prefetched = _exec_memory_search(
            db_path,
            query="",
            limit=50,
            created_after=essay_updated_at,
            entity_name=entity_name,
        )
        if prefetched.startswith("No memories found"):
            prefetched = "No new memories found since the essay was last updated."
        fmt["prefetched_memories"] = prefetched
        user_prompt = ESSAY_GENERATOR_USER_PATCH.format(**fmt)
    elif action == "regenerate":
        user_prompt = ESSAY_GENERATOR_USER_REGENERATE.format(**fmt)
    else:
        user_prompt = ESSAY_GENERATOR_USER_CREATE.format(**fmt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(max_rounds):
        content, tool_calls, _usage = llm_fn(messages, GENERATOR_TOOL_SCHEMAS)

        msg: dict = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)

        if tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError as e:
                    result = f"Error: malformed JSON: {e}"
                    args = {}
                else:
                    result = exec_generator_tool(name, args, db_path)
                log_fn(
                    f"  generate {entity_key} step {step} "
                    f"{name}({json.dumps(args)}) "
                    f"{'OK' if not result.startswith('Error') else 'ERR'}"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            continue

        matches = _GENERATE_COMPLETE_RE.findall(content or "")
        if matches:
            m_entity = matches[-1]
            con = sqlite3.connect(db_path)
            _ensure_essays_table(con)
            row = con.execute(
                "SELECT body FROM essays WHERE entity_key = ?",
                (entity_key,),
            ).fetchone()
            con.close()

            if not row:
                log_fn(f"  generate {entity_key} complete signal but no essay in DB")
                messages.append({"role": "user", "content":
                    f"You emitted <fold_complete> but no essay exists for "
                    f"{entity_key}. Use essay_edit to write it first."})
                continue

            err = validate_essay(row[0], essay_budget, ntokens_fn, known_ids)
            if err:
                log_fn(f"  generate {entity_key} validation fail: {err}")
                messages.append({"role": "user", "content":
                    f"VALIDATION FAILED: {err}. Fix the essay and re-emit "
                    f"<fold_complete>."})
                continue

            log_fn(f"  generate {entity_key} complete ({step + 1} steps)")
            return True, f"generated in {step + 1} steps"

        messages.append({"role": "user", "content":
            "Continue. Use memory_search/memory_get to retrieve memories, "
            "essay_edit to write, then emit <fold_complete> when done."})

    log_fn(f"  generate {entity_key} FAILED after {max_rounds} rounds")
    return False, f"failed after {max_rounds} rounds"


def _reset_dict_cache() -> None:
    """Reset the _DICT_WORDS cache. For test isolation only."""
    global _DICT_WORDS
    _DICT_WORDS = None
