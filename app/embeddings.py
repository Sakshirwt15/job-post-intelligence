"""Embedders. `st` = sentence-transformers (multilingual, Hinglish-friendly), `hash` = offline fallback."""
import re
import zlib

import numpy as np

from app import config


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class HashEmbedder:
    """Deterministic char-ngram hashing embedder. Quality low, but zero download - tests/dev ke liye."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = re.sub(r"\s+", " ", (t or "").lower())
            feats = re.findall(r"[a-z0-9]+", t)
            padded = f"  {t}  "
            feats += [padded[j : j + 3] for j in range(len(padded) - 2)]
            for f in feats:
                h = zlib.crc32(f.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model = SentenceTransformer(model_name or config.EMBED_MODEL)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder():
    if config.EMBED_BACKEND == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder()
