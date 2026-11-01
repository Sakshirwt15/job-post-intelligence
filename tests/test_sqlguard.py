import pytest

from app.sqlguard import SQLGuardError, clean_sql, ensure_limit, validate_select


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM jobs",
    "select company, role from jobs where location = 'Pune' and skills like '%Python%'",
    "WITH t AS (SELECT * FROM jobs) SELECT COUNT(*) FROM t",
    "SELECT company FROM jobs WHERE role LIKE '%update%'",  # keyword inside a string literal is fine
    "SELECT created_at FROM jobs",
])
def test_valid_queries(sql):
    assert validate_select(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM jobs",
    "DROP TABLE posts",
    "SELECT * FROM jobs; DROP TABLE posts",
    "SELECT * FROM posts",
    "SELECT * FROM llm_logs",
    "SELECT * FROM sqlite_master",
    "UPDATE jobs SET company='x'",
    "PRAGMA table_info(jobs)",
    "SELECT * FROM jobs -- comment",
    "SELECT * FROM other_table",
    "SELECT 1",
    "",
])
def test_blocked_queries(sql):
    with pytest.raises(SQLGuardError):
        validate_select(sql)


def test_clean_sql_strips_fences_and_semicolon():
    assert clean_sql("```sql\nSELECT 1 FROM jobs;\n```") == "SELECT 1 FROM jobs"


def test_ensure_limit():
    assert ensure_limit("SELECT * FROM jobs LIMIT 5") == "SELECT * FROM jobs LIMIT 5"
    assert ensure_limit("SELECT * FROM jobs").endswith("LIMIT 200")
