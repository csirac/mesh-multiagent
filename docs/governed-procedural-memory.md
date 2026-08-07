# Governed Procedural Memory

Mesh can preserve verified operational procedures as versioned YAML skill
cards. The feature is deliberately governed: an agent may draft a card, but a
human must approve it before the card becomes active.

## 1. Purpose

A card captures one narrow, repeatable procedure together with the conditions
under which it applies, its required invariants, verification steps, rollback
guidance, and evidence. It is advisory context, never independent authority to
perform an action outside the current request.

## 2. Design Principles

- Keep cards small, evidence-backed, and specific to an observable trigger.
- Preserve source fingerprints so a changed procedure is not reused silently.
- Separate drafting from activation; the authoring agent cannot approve itself.
- Treat a contradiction in required preconditions as disqualifying.

## 3. Card Schema

Every card has `schema_version`, `id`, `version`, `status`, `owner_agent`,
`purpose`, `triggers`, `preconditions`, `authority`, `procedure_source`,
`required_invariants`, `verification`, `rollback`, `caveats`, `evidence`,
`proposed_by`, `approved_by`, `approved_at`, and `outcomes` fields.

`procedure_source` entries name a stable artifact and its approved SHA-256
fingerprint. `caveats` must record the trace assessment plus non-empty lists of
pitfalls, unverified steps, and unexplored alternatives. A missing or stale
source fingerprint prevents retrieval.

## 4. Directory Layout

The default store is `~/.mesh/skills/<agent>/`. Active cards live directly
under that directory. Drafts are written only to `.proposals/`; historical
revisions are retained under `.history/`. The public example is
`docs/examples/skill_cards/example/model-service-recovery.yaml`.

## 5. Lifecycle

- `proposed`: schema-valid draft, never retrieved or injected.
- `active`: independently approved by `user:approver` with a timestamp; may be
  selected as advisory context.
- `retired`: retained for audit history, never selected.

Only a human changes a card from `proposed` to `active`. Each outcome is an
append-only receipt so use of a card is reviewable.

## 6. Selection

Retrieval uses lexical trigger matching plus typed preconditions. Contradicted
required preconditions disqualify a card before positive matches are scored.
At most three compatible cards are injected. A card cannot alter configured
models, credentials, filesystem scope, or user authority.

## 10. Governance and Safety

`skill_draft` packages explicitly named source files and an already-completed
worker trace into a staging task. The worker submits its draft through
`scripts/write_skill_proposal.py`; that tool validates the complete card and
writes only a proposed card. It cannot activate a card or mutate the active
index. A reviewer must verify every claimed invariant and rollback step against
the supplied evidence before activation.

## 11. Public Example

The model-service recovery example is intentionally generic. Replace its
placeholder service identity, approved sources, and probes with locally
verified values before proposing a real card. Never put credentials, private
hosts, or private operational history into a public example.
