"""Embedding-based dedup/clustering (pure numpy)."""
import numpy as np

from app.textutil import same_company


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = v / norms
    return v @ v.T


def cluster_vectors(vectors: np.ndarray, companies: list[str], threshold: float = 0.88) -> list[int]:
    """Union-find clustering. Do posts tab same cluster me jab cosine >= threshold AND company same.
    Return: har index ke liye cluster representative index (cluster ka sabse chhota index)."""
    n = len(companies)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if n:
        sims = cosine_matrix(vectors)
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= threshold and same_company(companies[i], companies[j]):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n)]
