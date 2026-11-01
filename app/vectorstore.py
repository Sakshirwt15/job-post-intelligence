"""Vector store: ChromaDB (persistent) ya in-memory numpy (tests). Same interface."""
import numpy as np

from app import config


def _match(meta: dict, where: dict | None) -> bool:
    for key, cond in (where or {}).items():
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, ref in cond.items():
                if val is None:
                    return False
                if op == "$gte" and not val >= ref:
                    return False
                if op == "$gt" and not val > ref:
                    return False
                if op == "$lte" and not val <= ref:
                    return False
                if op == "$lt" and not val < ref:
                    return False
        elif val != cond:
            return False
    return True


class InMemoryVectorStore:
    def __init__(self):
        self.items: dict[str, tuple[np.ndarray, str, dict]] = {}

    def upsert(self, post_id: int, text: str, embedding, metadata: dict) -> None:
        self.items[str(post_id)] = (np.asarray(embedding, dtype=np.float32), text, dict(metadata))

    def query(self, embedding, n: int = 10, where: dict | None = None):
        q = np.asarray(embedding, dtype=np.float32)
        scored = []
        for pid, (vec, _t, meta) in self.items.items():
            if _match(meta, where):
                scored.append((int(pid), float(np.dot(q, vec) / ((np.linalg.norm(q) * np.linalg.norm(vec)) or 1.0)), meta))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def update_metadata(self, post_id: int, metadata: dict) -> None:
        vec, text, meta = self.items[str(post_id)]
        meta.update(metadata)

    def count(self) -> int:
        return len(self.items)

    def reset(self) -> None:
        self.items.clear()


class ChromaVectorStore:
    def __init__(self, path=None, name: str = "jobs"):
        import chromadb  # lazy

        self.client = chromadb.PersistentClient(path=str(path or config.CHROMA_DIR))
        self.name = name
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _where(where: dict | None):
        if not where:
            return None
        conds = [{k: v} for k, v in where.items()]
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def upsert(self, post_id: int, text: str, embedding, metadata: dict) -> None:
        meta = {k: v for k, v in metadata.items() if v is not None}  # Chroma None allow nahi karta
        self.col.upsert(ids=[str(post_id)], embeddings=[np.asarray(embedding, dtype=float).tolist()],
                        documents=[text], metadatas=[meta])

    def query(self, embedding, n: int = 10, where: dict | None = None):
        total = self.col.count()
        if total == 0:
            return []
        res = self.col.query(query_embeddings=[np.asarray(embedding, dtype=float).tolist()],
                             n_results=min(n, total), where=self._where(where))
        ids, dists, metas = res["ids"][0], res["distances"][0], res["metadatas"][0]
        return [(int(i), 1.0 - float(d), m) for i, d, m in zip(ids, dists, metas)]

    def update_metadata(self, post_id: int, metadata: dict) -> None:
        got = self.col.get(ids=[str(post_id)])
        if got["ids"]:
            merged = {**(got["metadatas"][0] or {}), **metadata}
            self.col.update(ids=[str(post_id)], metadatas=[merged])

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})


def get_vectorstore():
    if config.VECTOR_BACKEND == "memory":
        return InMemoryVectorStore()
    return ChromaVectorStore()
