"""English/Hinglish question -> SQL (sirf SELECT, error par 3 retries)."""
from app import config
from app.db import Database
from app.llm import LLMClient
from app.prompts import load_prompt
from app.sqlguard import clean_sql, ensure_limit, validate_select

SYSTEM = "You are an expert SQLite analyst. Output only one SQL SELECT statement."


def ask_sql(db: Database, llm: LLMClient, question: str, prompt_version: str | None = None,
            max_retries: int | None = None) -> dict:
    version = prompt_version or config.SQL_PROMPT_VERSION
    max_retries = config.MAX_SQL_RETRIES if max_retries is None else max_retries
    base = load_prompt(version).replace("{{QUESTION}}", question.strip())
    prompt, errors = base, []

    for attempt in range(1, max_retries + 2):  # 1 try + max_retries retries
        raw_sql = ""
        try:
            out = llm.generate(prompt, system=SYSTEM, purpose="text2sql", prompt_version=version)
            raw_sql = clean_sql(out.text)
            sql = validate_select(raw_sql)
            cols, rows = db.run_select(ensure_limit(sql))
            return {"question": question, "sql": sql, "columns": cols, "rows": rows,
                    "attempts": attempt, "errors": errors, "success": True}
        except Exception as e:  # SQLGuardError, sqlite3.Error, LLMError
            errors.append(f"{type(e).__name__}: {e}")
            prompt = (base + f"\n\nYour previous SQL:\n{raw_sql or '(none)'}\n"
                      f"It failed with: {e}\nReturn a corrected single SELECT statement only.")
    return {"question": question, "sql": None, "columns": [], "rows": [], "attempts": max_retries + 1,
            "errors": errors, "success": False}
