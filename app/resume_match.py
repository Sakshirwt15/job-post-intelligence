"""Resume upload -> har job ke liye match score = 0.6*embedding sim + 0.4*keyword overlap."""
import io

import numpy as np

from app.matching import (chunk_text, combine_score, keyword_overlap, scale_similarity, top_chunk_similarity)
from app.textutil import job_to_text


def extract_resume_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def match_resume(svc, resume_text: str, top_n: int = 20, w_embed: float = 0.6, w_kw: float = 0.4) -> list[dict]:
    jobs = svc.db.list_posts(limit=5000, include_duplicates=False)
    if not jobs or not resume_text.strip():
        return []
    cvecs = svc.embedder.embed(chunk_text(resume_text))
    jvecs = svc.embedder.embed([job_to_text(j) for j in jobs])
    sims = np.asarray(jvecs) @ np.asarray(cvecs).T  # (jobs, resume_chunks)

    results = []
    for i, job in enumerate(jobs):
        cos = top_chunk_similarity(sims[i])
        emb_score = scale_similarity(cos)
        kw_score, matched, missing = keyword_overlap(job["skills"], resume_text)
        results.append({
            "post_id": job["id"], "company": job["company"], "role": job["role"], "location": job["location"],
            "score": round(100 * combine_score(emb_score, kw_score, w_embed, w_kw), 1),
            "embedding_similarity": round(cos, 3),
            "keyword_overlap": None if kw_score is None else round(kw_score, 2),
            "matched_skills": matched, "missing_skills": missing,
            "apply_link": job["apply_link"] or job["apply_email"],
        })
    results.sort(key=lambda r: -r["score"])
    return results[:top_n]
