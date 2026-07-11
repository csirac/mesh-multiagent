"""
Embedding client for the memory system.

Wraps OpenAI text-embedding-3-small by default. Configurable backend/model
for future local model support.
"""

import logging
import os

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Async embedding generation via OpenAI or local models."""

    def __init__(
        self,
        backend: str = "openai",
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self._backend = backend
        self._model = model

        if backend == "openai":
            self._client = AsyncOpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            )
        elif backend == "local":
            raise NotImplementedError("Local embedding backend not yet implemented")
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        if self._backend == "openai":
            response = await self._client.embeddings.create(
                input=text,
                model=self._model,
            )
            return response.data[0].embedding
        raise NotImplementedError(f"Backend {self._backend} not supported")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call. Returns list of embeddings."""
        if not texts:
            return []
        if self._backend == "openai":
            response = await self._client.embeddings.create(
                input=texts,
                model=self._model,
            )
            # Sort by index to maintain input order
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]
        raise NotImplementedError(f"Backend {self._backend} not supported")

    async def embed_to_array(self, text: str) -> np.ndarray:
        """Embed text and return as numpy array."""
        emb = await self.embed(text)
        return np.array(emb, dtype=np.float32)

    async def embed_batch_to_arrays(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts and return as numpy arrays."""
        embeddings = await self.embed_batch(texts)
        return [np.array(e, dtype=np.float32) for e in embeddings]
