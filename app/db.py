"""SQLite storage (stdlib sqlite3). Postgres/Supabase migration ke liye README ka roadmap dekho."""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    company TEXT NOT NULL,
    role TEXT NOT NULL,
    employment_type TEXT,
    location TEXT,
    is_remote INTEGER DEFAULT 0,
    stipend_min INTEGER,
    stipend_max INTEGER,
    batch_json TEXT DEFAULT '[]',
    skills_json TEXT DEFAULT '[]',
    apply_link TEXT,
    apply_email TEXT,
    prompt_version TEXT,
    cluster_id INTEGER,
    duplicate_of INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_hash ON posts(content_hash);
CREATE INDEX IF NOT EXISTS idx_posts_cluster ON posts(cluster_id);

CREATE TABLE IF NOT EXISTS llm_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    purpose TEXT,
    prompt_version TEXT,
    prompt TEXT,
    response TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    success INTEGER,
    error TEXT
);

DROP VIEW IF EXISTS jobs;
CREATE VIEW jobs AS
SELECT
    id, company, role, employment_type, location, is_remote,
    stipend_min, stipend_max,
    (SELECT group_concat(value, ',') FROM json_each(posts.batch_json)) AS batch_years,
    (SELECT group_concat(value, ', ') FROM json_each(posts.skills_json)) AS skills,
    apply_link, apply_email, cluster_id,
    (SELECT COUNT(*) FROM posts p2 WHERE p2.cluster_id = posts.cluster_id) AS times_posted,
    created_at
FROM posts
WHERE duplicate_of IS NULL;
"""


def _authorizer(action, arg1, arg2, dbname, source):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    rec = getattr(sqlite3, "SQLITE_RECURSIVE", None)
    if rec is not None:
        allowed.add(rec)
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---------- posts ----------
    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        d["batch"] = json.loads(d.pop("batch_json") or "[]")
        d["skills"] = json.loads(d.pop("skills_json") or "[]")
        d["is_remote"] = bool(d["is_remote"])
        return d

    def insert_post(self, raw_text: str, content_hash: str, job: dict, prompt_version: str | None,
                    duplicate_of: int | None = None, cluster_id: int | None = None) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO posts (raw_text, content_hash, company, role, employment_type, location,
                   is_remote, stipend_min, stipend_max, batch_json, skills_json, apply_link, apply_email,
                   prompt_version, duplicate_of, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (raw_text, content_hash, job["company"], job["role"], job.get("employment_type"),
                 job.get("location"), int(bool(job.get("is_remote"))), job.get("stipend_min"),
                 job.get("stipend_max"), json.dumps(job.get("batch") or []),
                 json.dumps(job.get("skills") or []), job.get("apply_link"), job.get("apply_email"),
                 prompt_version, duplicate_of, _now()),
            )
            pid = cur.lastrowid
            c.execute("UPDATE posts SET cluster_id=? WHERE id=?", (cluster_id if cluster_id is not None else pid, pid))
        return pid

    def get_post(self, post_id: int) -> dict | None:
        with self._conn() as c:
            return self._decode(c.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone())

    def find_by_hash(self, content_hash: str) -> dict | None:
        with self._conn() as c:
            return self._decode(c.execute("SELECT * FROM posts WHERE content_hash=? LIMIT 1", (content_hash,)).fetchone())

    def list_posts(self, limit: int = 200, offset: int = 0, include_duplicates: bool = True) -> list[dict]:
        q = "SELECT * FROM posts" + ("" if include_duplicates else " WHERE duplicate_of IS NULL")
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        with self._conn() as c:
            return [self._decode(r) for r in c.execute(q, (limit, offset)).fetchall()]

    def get_posts(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM posts WHERE id IN ({marks})", ids).fetchall()
        by_id = {r["id"]: self._decode(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def set_cluster(self, post_id: int, cluster_id: int, duplicate_of: int | None) -> None:
        with self._conn() as c:
            c.execute("UPDATE posts SET cluster_id=?, duplicate_of=? WHERE id=?", (cluster_id, duplicate_of, post_id))

    def count_posts(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    # ---------- logs ----------
    def log_llm_call(self, **kw) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO llm_logs (ts, provider, model, purpose, prompt_version, prompt, response,
                   latency_ms, prompt_tokens, completion_tokens, success, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (_now(), kw.get("provider"), kw.get("model"), kw.get("purpose"), kw.get("prompt_version"),
                 kw.get("prompt"), kw.get("response"), kw.get("latency_ms"), kw.get("prompt_tokens", 0),
                 kw.get("completion_tokens", 0), int(bool(kw.get("success"))), kw.get("error")),
            )

    def fetch_logs(self, limit: int = 5000) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM llm_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    # ---------- read-only SQL for text-to-SQL ----------
    def run_select(self, sql: str, timeout_s: float = 5.0, max_rows: int = 500):
        c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=timeout_s)
        try:
            c.execute("PRAGMA query_only = ON")
            deadline = time.monotonic() + timeout_s
            c.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
            c.set_authorizer(_authorizer)
            cur = c.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [tuple(r) for r in cur.fetchmany(max_rows)]
            return cols, rows
        finally:
            c.close()
