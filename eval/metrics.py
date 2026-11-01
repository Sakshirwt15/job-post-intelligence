"""Eval metrics (pure python - pydantic/LLM ki zarurat nahi)."""
import itertools
import re

FIELDS = ["company", "role", "employment_type", "batch", "stipend_min", "stipend_max",
          "location", "is_remote", "skills", "apply_link", "apply_email"]


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#. ]", " ", str(s or "").lower())).strip()


def _tok(s):
    return set(re.findall(r"[a-z0-9]+", str(s or "").lower()))


def _f1(pred, gold):
    p, g = {_norm(x) for x in pred or []}, {_norm(x) for x in gold or []}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def _jaccard_ok(a, b, thr=0.67):
    ta, tb = _tok(a), _tok(b)
    return float(bool(ta | tb) and len(ta & tb) / len(ta | tb) >= thr)


def _url(u):
    return (u or "").strip().rstrip("/").lower() or None


def field_scores(pred: dict | None, gold: dict) -> dict:
    """Har field ke liye 0..1 score. pred None (extraction fail) => sab 0."""
    if pred is None:
        return {f: 0.0 for f in FIELDS}
    return {
        "company": float(_norm(pred.get("company")) == _norm(gold.get("company"))),
        "role": _jaccard_ok(pred.get("role"), gold.get("role")),
        "employment_type": float(pred.get("employment_type") == gold.get("employment_type")),
        "batch": float(sorted(pred.get("batch") or []) == sorted(gold.get("batch") or [])),
        "stipend_min": float(pred.get("stipend_min") == gold.get("stipend_min")),
        "stipend_max": float(pred.get("stipend_max") == gold.get("stipend_max")),
        "location": float(_norm(pred.get("location")) == _norm(gold.get("location"))),
        "is_remote": float(bool(pred.get("is_remote")) == bool(gold.get("is_remote"))),
        "skills": _f1(pred.get("skills"), gold.get("skills")),
        "apply_link": float(_url(pred.get("apply_link")) == _url(gold.get("apply_link"))),
        "apply_email": float(_norm(pred.get("apply_email")) == _norm(gold.get("apply_email"))),
    }


def aggregate(all_scores: list[dict]) -> dict:
    n = len(all_scores)
    per_field = {f: round(sum(s[f] for s in all_scores) / n, 4) for f in FIELDS} if n else {}
    overall = round(sum(per_field.values()) / len(per_field), 4) if per_field else 0.0
    exact = round(sum(all(v >= 0.99 for v in s.values()) for s in all_scores) / n, 4) if n else 0.0
    return {"per_field": per_field, "overall": overall, "all_fields_correct": exact}


# ---------- text-to-SQL result comparison ----------
def _cell(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _rows(rows):
    return sorted((tuple(_cell(c) for c in r) for r in rows), key=lambda r: [str(x) for x in r])


def results_match(gold_rows, pred_rows) -> bool:
    """Exact multiset match, ya predicted me extra columns ho to gold columns ka koi row-aligned projection match kare."""
    g = _rows(gold_rows)
    p = _rows(pred_rows)
    if g == p:
        return True
    if len(gold_rows) != len(pred_rows) or not gold_rows or not pred_rows:
        return False
    ng, npred = len(gold_rows[0]), len(pred_rows[0])
    if npred <= ng or ng > 4:
        return False
    for cols in itertools.permutations(range(npred), ng):
        if _rows([tuple(r[c] for c in cols) for r in pred_rows]) == g:
            return True
    return False
