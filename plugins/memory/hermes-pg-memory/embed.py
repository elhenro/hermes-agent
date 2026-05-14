"""Embedding generation via Ollama API."""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_DIMS = 768


class Embedder:
    """Generate embeddings via Ollama's /api/embed endpoint.

    Falls back to /api/embeddings (singular) for older Ollama versions.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dims: int = DEFAULT_DIMS,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dims = dims
        self._embed_url = f"{self.base_url}/api/embed"
        self._embeddings_url = f"{self.base_url}/api/embeddings"

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed a single text. Returns vec or None on failure."""
        try:
            resp = requests.post(
                self._embed_url,
                json={"model": self.model, "input": text},
                timeout=30,
            )
            if resp.status_code == 404:
                # legacy fallback
                resp = requests.post(
                    self._embeddings_url,
                    json={"model": self.model, "prompt": text},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("embedding")
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings") or []
            if embeddings:
                return embeddings[0]
            return None
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []
        try:
            resp = requests.post(
                self._embed_url,
                json={"model": self.model, "input": texts},
                timeout=60,
            )
            if resp.status_code == 404:
                # legacy endpoint has no batch; fall back to per-item
                return [self.embed(t) for t in texts]
            resp.raise_for_status()
            data = resp.json()
            return data.get("embeddings", [])
        except Exception as e:
            logger.warning("Batch embedding failed: %s", e)
            return [None] * len(texts)

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
