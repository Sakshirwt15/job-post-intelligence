"""RAG search: Hinglish query -> (english query + filters) -> filtered vector search -> grounded answer."""
import json
import re

from app import config
from app.prompts import load_prompt
from app.textutil import parse_json_loose


def _int_or_none(v):
    try:
        return int(float(v)) if v not in (None, "", "null") else None
    except (TypeError, ValueError):
        return None


def parse_query(svc, query: str) -> dict:
    """LLM se query ko {english_query, location, min_stipend, employment_type, skills} me todo. Fail ho to raw query."""
    try:
        prompt = load_prompt(config.QUERY_PARSE_PROMPT_VERSION).replace("{{QUERY}}", query)
        out = svc.llm.generate(prompt, purpose="query_parse", prompt_version=config.QUERY_PARSE_PROMPT_VERSION, json_mode=True)
        data = parse_json_loose(out.text)
    except Exception:  # noqa: BLE001 - search ko LLM failure se marna nahi chahiye
        data = {}
    emp = str(data.get("employment_type") or "").lower()
    skills = data.get("skills") if isinstance(data.get("skills"), list) else []
    return {
        "english_query": str(data.get("english_query") or query),
        "location": (str(data["location"]).strip() or None) if data.get("location") else None,
        "min_stipend": _int_or_none(data.get("min_stipend")),
        "employment_type": emp if emp in {"internship", "full-time", "contract"} else None,
        "skills": [str(s) for s in skills][:8],
    }


def _where(parsed: dict) -> dict:
    where = {"is_duplicate": 0}
    if parsed["location"]:
        where["location_lc"] = parsed["location"].lower()
    if parsed["min_stipend"]:
        where["stipend_top"] = {"$gte": parsed["min_stipend"]}
    if parsed["employment_type"]:
        where["employment_type"] = parsed["employment_type"]
    return where


def _answer(svc, query: str, jobs: list[dict]) -> str:
    if not jobs:
        return "Koi matching job nahi mili."
    compact = [{"id": j["id"], "company": j["company"], "role": j["role"], "type": j["employment_type"],
                "location": j["location"] or ("Remote" if j["is_remote"] else None),
                "stipend": f"{j['stipend_min'] or ''}-{j['stipend_max'] or ''}", "skills": j["skills"],
                "batch": j["batch"], "apply": j["apply_link"] or j["apply_email"]} for j in jobs]
    prompt = (load_prompt(config.ANSWER_PROMPT_VERSION)
              .replace("{{QUESTION}}", query).replace("{{JOBS}}", json.dumps(compact, ensure_ascii=False, indent=1)))
    try:
        return svc.llm.generate(prompt, purpose="rag_answer", prompt_version=config.ANSWER_PROMPT_VERSION).text.strip()
    except Exception:  # noqa: BLE001
        return "\n".join(f"- [{j['id']}] {j['role']} @ {j['company']} ({j['location'] or 'Remote'})" for j in jobs)


def search(svc, query: str, k: int = 5) -> dict:
    parsed = parse_query(svc, query)
    qvec = svc.embedder.embed([parsed["english_query"]])[0]
    relaxed = False
    hits = svc.vstore.query(qvec, k, where=_where(parsed))
    if not hits and len(_where(parsed)) > 1:  # filters bahut tight - hata ke dobara try
        hits, relaxed = svc.vstore.query(qvec, k, where={"is_duplicate": 0}), True
    scores = {pid: sim for pid, sim, _ in hits}
    jobs = svc.db.get_posts([pid for pid, _s, _m in hits])
    return {
        "query": query, "parsed": parsed, "filters_relaxed": relaxed,
        "answer": _answer(svc, query, jobs),
        "sources": [{**j, "similarity": round(scores.get(j["id"], 0.0), 3)} for j in jobs],
    }
