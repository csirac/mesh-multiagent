#!/usr/bin/env python3
"""
Retrofit retrieval keys for existing memory entries.

For each memory in an agent's pool, feeds summary+reflection to an LLM
to generate a concise task descriptor (retrieval key), then batch-embeds
all keys and updates the DB.

Usage:
  # Dry-run: see what keys would be generated (processes 3 entries)
  python -m mesh.scripts.retrofit_retrieval_keys --nickname bob --dry-run

  # Live: generate and store retrieval keys for all entries
  python -m mesh.scripts.retrofit_retrieval_keys --nickname bob

  # Use a specific LLM backend
  python -m mesh.scripts.retrofit_retrieval_keys --nickname bob --backend claude-code --model opus
"""

import argparse
import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mesh.memory.store import MemoryEntry, MemoryStore, _serialize_embedding
from mesh.memory.system import _extract_tag
from mesh.memory.embeddings import EmbeddingClient
from mesh.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)

RETRIEVAL_KEY_PROMPT = """Given this memory summary and reflection from a past task, write a 1-2 sentence description of what the task or conversation was about.

Be specific — name technologies, systems, files, and concepts involved.
Focus on the primary topic or task. If the session covered multiple topics, describe the most significant one.
This will be used as a search key, so include terms a future query would match.

<summary>{summary}</summary>

<reflection>{reflection}</reflection>

Respond with exactly:

<retrieval_key>your description here</retrieval_key>"""


async def retrofit(args: argparse.Namespace) -> None:
    """Main entry point: generate retrieval keys and re-embed."""
    store = MemoryStore(args.nickname)
    entries = store.load()

    if not entries:
        print(f"No memories found for '{args.nickname}'. Nothing to do.")
        store.close()
        return

    # Filter to entries that need retrofit (empty retrieval_key)
    needs_retrofit = [e for e in entries if not e.retrieval_key]
    print(f"Found {len(entries)} memories for '{args.nickname}', "
          f"{len(needs_retrofit)} need retrieval keys")

    if not needs_retrofit:
        print("All entries already have retrieval keys. Nothing to do.")
        store.close()
        return

    if args.dry_run:
        # Show a sample of entries that would be processed
        sample = needs_retrofit[:3]
        print(f"\nDry run — showing {len(sample)} sample entries:\n")
        for entry in sample:
            print(f"  ID: {entry.id}")
            print(f"  Summary: {entry.summary[:120]}...")
            print(f"  Trigger: {entry.trigger[:80]}...")
            print()
        print(f"Would process {len(needs_retrofit)} entries total.")
        store.close()
        return

    # Set up LLM client
    backend = args.backend or "openai"
    model = args.model or "gpt-5.1"
    llm_config = LLMConfig.from_env(backend=backend)
    llm_config.model = model
    llm_config.max_tokens = 256
    llm_config.temperature = 0.3

    embedder = EmbeddingClient(
        backend="openai",
        model="text-embedding-3-small",
    )

    generated_keys: list[tuple[MemoryEntry, str]] = []
    errors = 0

    async with LLMClient(llm_config) as llm_client:
        for i, entry in enumerate(needs_retrofit):
            print(f"[{i+1}/{len(needs_retrofit)}] {entry.id}: ", end="", flush=True)

            prompt = RETRIEVAL_KEY_PROMPT.format(
                summary=entry.summary,
                reflection=entry.reflection[:2000],
            )

            try:
                response = await llm_client.complete(prompt)
                retrieval_key = _extract_tag(response, "retrieval_key")
            except Exception:
                logger.error(f"LLM call failed for {entry.id}", exc_info=True)
                print("ERROR (LLM)")
                errors += 1
                continue

            if not retrieval_key:
                # Fallback to summary
                retrieval_key = entry.summary
                print(f"fallback to summary")
            else:
                key_preview = retrieval_key[:80].replace("\n", " ")
                print(f"\"{key_preview}\"")

            generated_keys.append((entry, retrieval_key))

    if not generated_keys:
        print(f"\nNo retrieval keys generated. {errors} errors.")
        store.close()
        return

    # Batch embed all retrieval keys
    print(f"\nBatch embedding {len(generated_keys)} retrieval keys...")
    keys_text = [key for _, key in generated_keys]
    try:
        key_embeddings = await embedder.embed_batch_to_arrays(keys_text)
    except Exception:
        logger.error("Batch embedding failed", exc_info=True)
        print("ERROR: Batch embedding failed. No changes written.")
        store.close()
        return

    # Update DB entries
    print(f"Updating {len(generated_keys)} entries in DB...")
    conn = store._conn
    for (entry, key), emb in zip(generated_keys, key_embeddings):
        conn.execute(
            "UPDATE memories SET retrieval_key = ?, retrieval_key_embedding = ? "
            "WHERE id = ?",
            (key, _serialize_embedding(emb), entry.id),
        )
    conn.commit()

    print(f"\nDone!")
    print(f"  Updated: {len(generated_keys)}")
    print(f"  Errors: {errors}")
    print(f"  Skipped (already had key): {len(entries) - len(needs_retrofit)}")

    store.close()


def main():
    parser = argparse.ArgumentParser(
        description="Retrofit retrieval keys for existing memory entries"
    )
    parser.add_argument("--nickname", required=True,
                        help="Agent nickname (e.g., bob, claude-sobek)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without making changes")
    parser.add_argument("--backend", type=str, default=None,
                        help="LLM backend for key generation (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model for key generation (default: gpt-5.1)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    asyncio.run(retrofit(args))


if __name__ == "__main__":
    main()
