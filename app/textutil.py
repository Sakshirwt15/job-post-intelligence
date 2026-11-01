"""Chhote pure helpers (no pydantic, no network) - easily testable."""
import hashlib
import json
import re

_COMPANY_SUFFIXES = {
    "pvt", "ltd", "private", "limited", "inc", "llp", "llc", "corp", "co",
    "technologies", "technology", "tech", "solutions", "labs", "systems",
}


def normalize_company(name: str | None) -> str:
    if not name:
        return ""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    core = [t for t in tokens if t not in _COMPANY_SUFFIXES]
    return "".join(core or tokens)


def same_company(a: str | None, b: str | None) -> bool:
    na, nb = normalize_company(a), normalize_company(b)
    if not na or not nb:
        return False
    return na == nb or (len(min(na, nb, key=len)) >= 4 and (na in nb or nb in na))


def tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def job_to_text(job: dict) -> str:
    """Embedding ke liye job ka compact text (dict se - DB row ya JobPost.model_dump())."""
    parts = [f"{job.get('role', '')} at {job.get('company', '')}"]
    if job.get("employment_type") and job["employment_type"] != "unknown":
        parts.append(str(job["employment_type"]))
    if job.get("location"):
        parts.append(f"location {job['location']}")
    if job.get("is_remote"):
        parts.append("remote")
    if job.get("skills"):
        parts.append("skills: " + ", ".join(job["skills"]))
    if job.get("batch"):
        parts.append("batch " + "/".join(str(b) for b in job["batch"]))
    if job.get("stipend_min") or job.get("stipend_max"):
        parts.append(f"stipend {job.get('stipend_min') or ''}-{job.get('stipend_max') or ''} per month")
    return ". ".join(parts)


def parse_json_loose(text: str) -> dict:
    """LLM output se JSON object nikalta hai (code fences / extra text tolerate karta hai)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        obj = json.loads(t[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Model output JSON is not an object")
    return obj


def content_hash(raw_text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", raw_text.strip().lower()).encode()).hexdigest()
