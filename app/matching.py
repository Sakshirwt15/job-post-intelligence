"""Resume <-> job match: embedding similarity + keyword overlap (pure functions)."""
import re

import numpy as np

SKILL_ALIASES = [
    {"javascript", "js"}, {"machine learning", "ml"}, {"deep learning", "dl"},
    {"natural language processing", "nlp"}, {"scikit-learn", "sklearn", "scikit learn"},
    {"postgresql", "postgres"}, {"react", "reactjs", "react.js"},
    {"node.js", "nodejs", "node js"}, {"golang", "go lang"}, {"tensorflow", "tf"},
    {"large language models", "llm", "llms"}, {"power bi", "powerbi"}, {"kubernetes", "k8s"},
    {"amazon web services", "aws"}, {"artificial intelligence", "ai"},
]


def _variants(skill: str) -> set[str]:
    s = skill.lower().strip()
    for group in SKILL_ALIASES:
        if s in group:
            return set(group)
    return {s}


def skill_in_text(skill: str, text_lower: str) -> bool:
    for v in _variants(skill):
        if re.search(r"(?<![a-z0-9+#])" + re.escape(v) + r"(?![a-z0-9+#])", text_lower):
            return True
    return False


def keyword_overlap(job_skills: list[str], resume_text: str):
    """(score 0..1, matched, missing). Job me skills nahi hain to score None."""
    skills = [s for s in job_skills if s and s.strip()]
    if not skills:
        return None, [], []
    low = resume_text.lower()
    matched = [s for s in skills if skill_in_text(s, low)]
    missing = [s for s in skills if s not in matched]
    return len(matched) / len(skills), matched, missing


def scale_similarity(cos: float, lo: float = 0.15, hi: float = 0.65) -> float:
    """Raw cosine (jo aksar 0.2-0.6 rehta hai) ko 0..1 me stretch karo."""
    return float(min(1.0, max(0.0, (cos - lo) / (hi - lo))))


def combine_score(embed_score: float, kw_score: float | None, w_embed: float = 0.6, w_kw: float = 0.4) -> float:
    if kw_score is None:
        return embed_score
    return w_embed * embed_score + w_kw * kw_score


def chunk_text(text: str, words_per_chunk: int = 80) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def top_chunk_similarity(sim_row: np.ndarray, top: int = 2) -> float:
    return float(np.sort(sim_row)[::-1][:top].mean())
