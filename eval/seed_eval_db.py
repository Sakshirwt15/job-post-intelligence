"""labelled_posts.json se ek eval DB banata hai (LLM ke bina) - text-to-SQL eval ke liye."""
import json
from pathlib import Path

from app.db import Database
from app.textutil import content_hash

DATASET = Path(__file__).parent / "labelled_posts.json"


def build_db(db_path) -> Database:
    p = Path(db_path)
    if p.exists():
        p.unlink()
    db = Database(p)
    first_in_group: dict[int, int] = {}
    for item in json.loads(DATASET.read_text(encoding="utf-8")):
        g = item["group_id"]
        dup_of = first_in_group.get(g)
        cluster = None
        if dup_of:
            cluster = db.get_post(dup_of)["cluster_id"]
        pid = db.insert_post(item["text"], content_hash(item["text"]), item["labels"], "gold",
                             duplicate_of=dup_of, cluster_id=cluster)
        first_in_group.setdefault(g, pid)
    return db
