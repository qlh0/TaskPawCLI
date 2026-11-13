"""embedding.py — pluggable text embeddings for hybrid retrieval.

The default backend is fastembed (ONNX), which keeps the CLI free of torch.
Tests use :class:`HashEmbedder`, a deterministic bag-of-hashes embedder.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]")


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedBackend:
    """Lazy ONNX embedder; the model is only loaded on first use."""

    def __init__(
        self,
        *,
        model_name: str,
        dim: int,
        cache_dir: str,
        threads: int = 4,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._dim = int(dim)
        self._cache_dir = str(Path(cache_dir).expanduser())
        self._threads = max(1, int(threads))
        self._batch_size = max(1, int(batch_size))
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        model = self._ensure_model()
        vectors = [
            [float(value) for value in vector]
            for vector in model.embed(items, batch_size=self._batch_size)
        ]
        return vectors

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(
                    model_name=self._model_name,
                    cache_dir=self._cache_dir,
                    threads=self._threads,
                )
                detected = _detect_dim(self._model_name)
                if detected:
                    self._dim = detected
        return self._model


class HashEmbedder:
    """Deterministic embedder for offline tests (shared tokens ⇒ similar vectors)."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = max(8, int(dim))

    @property
    def model_name(self) -> str:
        return "hash"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dim
            for token in _TOKEN_RE.findall(str(text).lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self._dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                # Hash collisions can cancel out; keep a deterministic non-zero vector.
                bucket = hashlib.blake2b(str(text).encode("utf-8"), digest_size=4).digest()[0]
                vector[bucket % self._dim] = 1.0
                norm = 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def create_embedder(config) -> Embedder | None:
    """Build the configured embedder, or ``None`` when embeddings are off."""

    settings = config.embeddings
    if not settings.enabled:
        return None
    if settings.backend == "hash":
        return HashEmbedder(settings.dim)
    if settings.backend == "fastembed":
        return FastEmbedBackend(
            model_name=settings.model,
            dim=settings.dim,
            cache_dir=settings.cache_dir,
            threads=settings.threads,
            batch_size=settings.batch_size,
        )
    return None


def _detect_dim(model_name: str) -> int:
    try:
        from fastembed import TextEmbedding

        for item in TextEmbedding.list_supported_models():
            if item.get("model") == model_name:
                return int(item.get("dim") or 0)
    except Exception:  # noqa: BLE001 - dimension is only a hint
        return 0
    return 0
