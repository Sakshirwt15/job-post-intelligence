"""Ek jagah saari wiring: ingest (extract -> validate -> DB -> embed -> dedup) + shared by API & UI."""
from functools import lru_cache

import numpy as np

from app import config
from app.db import Database
from app.dedup import cluster_vectors
from app.embeddings import get_embedder
from app.extraction import ExtractionError, extract_job
from app.llm import LLMClient
from app.textutil import content_hash, job_to_text, same_company
from app.vectorstore import get_vectorstore


def vector_metadata(job: dict, is_duplicate: bool) -> dict:
    top = job.get("stipend_max") or job.get("stipend_min") or 0
    return {
        "is_duplicate": int(is_duplicate),
        "employment_type": job.get("employment_type") or "unknown",
        "location_lc": (job.get("location") or "").lower(),
        "stipend_top": int(top),
        "is_remote": int(bool(job.get("is_remote"))),
    }


class JobIntelligence:
    def __init__(self, db: Database, llm: LLMClient, embedder, vstore, dedup_threshold: float | None = None):
        self.db, self.llm, self.embedder, self.vstore = db, llm, embedder, vstore
        self.dedup_threshold = config.DEDUP_THRESHOLD if dedup_threshold is None else dedup_threshold

    # ---------- ingest ----------
    def ingest(self, raw_text: str, prompt_version: str | None = None) -> dict:
        raw_text = raw_text.strip()
        if len(raw_text) < 20:
            raise ValueError("Post text is too short")

        h = content_hash(raw_text)
        existing = self.db.find_by_hash(h)
        if existing:
            return {"status": "exact_duplicate", "post_id": existing["id"], "duplicate_of": existing["id"], "job": existing}

        result = extract_job(self.llm, raw_text, prompt_version)  # may raise ExtractionError
        job = result.job.model_dump()
        vec = self.embedder.embed([job_to_text(job)])[0]

        dup_id, cluster_id = None, None
        for pid, sim, _meta in self.vstore.query(vec, 5, where={"is_duplicate": 0}):
            other = self.db.get_post(pid)
            if other and sim >= self.dedup_threshold and same_company(job["company"], other["company"]):
                dup_id, cluster_id = other["id"], other["cluster_id"]
                break

        post_id = self.db.insert_post(raw_text, h, job, prompt_version or config.EXTRACT_PROMPT_VERSION,
                                      duplicate_of=dup_id, cluster_id=cluster_id)
        self.vstore.upsert(post_id, job_to_text(job), vec, vector_metadata(job, dup_id is not None))
        return {"status": "duplicate" if dup_id else "new", "post_id": post_id, "duplicate_of": dup_id,
                "attempts": result.attempts, "validation_errors": result.errors, "job": job}

    def ingest_many(self, posts: list[str]) -> list[dict]:
        out = []
        for p in posts:
            try:
                out.append(self.ingest(p))
            except (ExtractionError, ValueError) as e:
                out.append({"status": "failed", "error": str(e), "preview": p[:80]})
        return out

    # ---------- re-cluster everything ----------
    def recluster(self) -> dict:
        posts = self.db.list_posts(limit=100000)[::-1]  # oldest first
        if not posts:
            return {"posts": 0, "clusters": 0}
        vecs = self.embedder.embed([job_to_text(p) for p in posts])
        labels = cluster_vectors(np.asarray(vecs), [p["company"] for p in posts], self.dedup_threshold)
        for i, p in enumerate(posts):
            rep = posts[labels[i]]
            dup_of = None if labels[i] == i else rep["id"]
            self.db.set_cluster(p["id"], rep["id"], dup_of)
            self.vstore.upsert(p["id"], job_to_text(p), vecs[i], vector_metadata(p, dup_of is not None))
        return {"posts": len(posts), "clusters": len(set(labels))}

    # ---------- features (thin wrappers) ----------
    def search(self, query: str, k: int = 5) -> dict:
        from app.search import search
        return search(self, query, k)

    def ask_sql(self, question: str) -> dict:
        from app.text2sql import ask_sql
        return ask_sql(self.db, self.llm, question)

    def match_resume(self, resume_text: str, top_n: int = 20) -> list[dict]:
        from app.resume_match import match_resume
        return match_resume(self, resume_text, top_n)


@lru_cache(maxsize=1)
def get_service() -> JobIntelligence:
    db = Database(config.DB_PATH)
    return JobIntelligence(db, LLMClient(db=db), get_embedder(), get_vectorstore())
