"""External embedding provider with batching and instruction prefixes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .. import config
from ..provider import embed as provider_embed

log = logging.getLogger(__name__)


class EmbeddingUnavailableError(Exception):
    """Raised when all embedding providers fail."""


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    """Resolved embedding configuration."""

    model: str
    dimensions: int
    query_instruction: str
    doc_instruction: str
    batch_size: int


def _resolve_config() -> EmbedderConfig:
    return EmbedderConfig(
        model=config.EMBEDDING_MODEL,
        dimensions=config.EMBEDDING_DIMENSIONS,
        query_instruction=config.EMBEDDING_QUERY_INSTRUCTION,
        doc_instruction=config.EMBEDDING_DOC_INSTRUCTION,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )


DEFAULT_EMBEDDER_CONFIG = _resolve_config()


class ExternalEmbedder:
    """Embedding provider with batching and instruction-aware embeddings."""

    def __init__(self, cfg: EmbedderConfig = DEFAULT_EMBEDDER_CONFIG) -> None:
        self._cfg = cfg

    @property
    def dimensions(self) -> int:
        return self._cfg.dimensions

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with retrieval-optimized instruction."""
        prefixed = f"{self._cfg.query_instruction} {query}"
        result = self._embed_batched([prefixed])
        return result[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Batch embed documents with storage-optimized instruction."""
        if not documents:
            return []
        prefixed = [f"{self._cfg.doc_instruction} {doc}" for doc in documents]
        return self._embed_batched(prefixed)

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Embed with automatic sub-batching."""
        all_embeddings: list[list[float]] = []
        batch_size = self._cfg.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._call_provider(batch)
            all_embeddings.extend(embeddings)

        if len(all_embeddings) != len(texts):
            raise EmbeddingUnavailableError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}"
            )
        return all_embeddings

    def _call_provider(self, texts: list[str]) -> list[list[float]]:
        """Call the unified provider embedding endpoint."""
        try:
            return provider_embed(
                model=self._cfg.model,
                texts=texts,
                dimensions=self._cfg.dimensions,
            )
        except Exception as exc:
            log.error("Embedding request failed: %s", exc)
            raise EmbeddingUnavailableError(str(exc)) from exc
