"""Text-to-SQL safety: sirf single read-only SELECT, sirf `jobs` view."""
import re

ALLOWED_TABLES = {"jobs"}
FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "attach", "detach",
    "pragma", "vacuum", "reindex", "analyze", "truncate", "grant", "revoke",
    "begin", "commit", "rollback", "savepoint", "load_extension",
]
FORBIDDEN_TABLES = ["posts", "llm_logs", "sqlite_master", "sqlite_schema", "sqlite_temp_master"]


class SQLGuardError(ValueError):
    pass


def _strip_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def clean_sql(raw: str) -> str:
    """Model output se SQL nikalo (fences hatao, trailing ; hatao)."""
    s = (raw or "").strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    return s.rstrip().rstrip(";").strip()


def validate_select(sql: str) -> str:
    """Valid hai to cleaned SQL return karo, warna SQLGuardError."""
    s = clean_sql(sql)
    if not s:
        raise SQLGuardError("Empty SQL")
    bare = _strip_literals(s)
    low = bare.lower()

    if ";" in bare:
        raise SQLGuardError("Multiple statements are not allowed")
    if "--" in bare or "/*" in bare:
        raise SQLGuardError("SQL comments are not allowed")
    if not re.match(r"^\s*(select|with)\b", low):
        raise SQLGuardError("Only SELECT queries are allowed")
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", low):
            raise SQLGuardError(f"Forbidden keyword: {kw}")
    for t in FORBIDDEN_TABLES:
        if re.search(rf"\b{t}\b", low):
            raise SQLGuardError(f"Table not allowed: {t}")

    cte_names = set(re.findall(r"\b([a-z_]\w*)\s+as\s*\(", low))
    used = set(re.findall(r"\b(?:from|join)\s+([a-z_]\w*)", low))
    bad = used - ALLOWED_TABLES - cte_names
    if bad:
        raise SQLGuardError(f"Unknown table(s): {', '.join(sorted(bad))}. Use only: jobs")
    if "jobs" not in used:
        raise SQLGuardError("Query must read from the `jobs` view")
    return s


def ensure_limit(sql: str, limit: int = 200) -> str:
    if re.search(r"\blimit\s+\d+\s*$", sql, flags=re.IGNORECASE):
        return sql
    return f"SELECT * FROM ({sql}) LIMIT {limit}"
