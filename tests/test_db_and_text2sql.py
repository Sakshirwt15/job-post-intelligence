import sqlite3

import pytest

from app.text2sql import ask_sql
from tests.conftest import FakeLLM

JOB = {"company": "Acme", "role": "ML Intern", "employment_type": "internship", "location": "Pune",
       "is_remote": False, "stipend_min": 20000, "stipend_max": 30000, "batch": [2025], "skills": ["Python", "SQL"],
       "apply_link": "https://x.io/a", "apply_email": None}


@pytest.fixture
def seeded(db):
    db.insert_post("raw1", "h1", JOB, "v")
    db.insert_post("raw2", "h2", {**JOB, "company": "Beta", "location": "Mumbai"}, "v")
    return db


def test_view_flattens_json(seeded):
    cols, rows = seeded.run_select("SELECT skills, batch_years FROM jobs WHERE company='Acme'")
    assert rows == [("Python, SQL", "2025")]


def test_duplicates_hidden_from_view(seeded):
    seeded.insert_post("raw3", "h3", JOB, "v", duplicate_of=1, cluster_id=1)
    assert seeded.run_select("SELECT COUNT(*) FROM jobs")[1] == [(2,)]
    assert seeded.run_select("SELECT times_posted FROM jobs WHERE id=1")[1] == [(2,)]


def test_readonly_connection_blocks_writes(seeded):
    with pytest.raises(sqlite3.Error):
        seeded.run_select("DELETE FROM posts")


def test_text2sql_happy_path(seeded):
    res = ask_sql(seeded, FakeLLM(["SELECT COUNT(*) FROM jobs WHERE location = 'Pune'"]), "how many in pune")
    assert res["success"] and res["rows"] == [(1,)] and res["attempts"] == 1


def test_text2sql_retries_after_forbidden_sql(seeded):
    llm = FakeLLM(["DELETE FROM jobs", "SELECT COUNT(*) FROM jobs"])
    res = ask_sql(seeded, llm, "count")
    assert res["success"] and res["attempts"] == 2 and "SQLGuardError" in res["errors"][0]


def test_text2sql_retries_after_sql_error(seeded):
    llm = FakeLLM(["SELECT nonexistent FROM jobs", "SELECT COUNT(*) FROM jobs"])
    assert ask_sql(seeded, llm, "count")["attempts"] == 2


def test_text2sql_gives_up_after_three_retries(seeded):
    res = ask_sql(seeded, FakeLLM(["DROP TABLE posts"]), "x")
    assert not res["success"] and res["attempts"] == 4 and len(res["errors"]) == 4
