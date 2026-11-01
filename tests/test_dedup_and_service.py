import numpy as np

from app.dedup import cluster_vectors
from app.embeddings import HashEmbedder
from tests.conftest import GOOD_JSON

REPOST = GOOD_JSON  # same extracted job from a differently worded post


def test_cluster_vectors_groups_same_company_only():
    v = np.array([[1, 0], [0.99, 0.05], [1, 0]], dtype=float)
    labels = cluster_vectors(v, ["Acme Pvt Ltd", "Acme", "Other Co"], threshold=0.9)
    assert labels == [0, 0, 2]


def test_ingest_new_then_semantic_duplicate(make_service):
    svc = make_service([GOOD_JSON])
    first = svc.ingest("Zephyr Labs hiring Data Analyst Intern in Pune 15-20k. Apply https://forms.gle/abc123")
    second = svc.ingest("🚀 Hiring! Data Analyst Intern @ Zephyr Labs, Pune. Stipend 15k-20k. forms.gle/abc123 apply now")
    assert first["status"] == "new"
    assert second["status"] == "duplicate" and second["duplicate_of"] == first["post_id"]
    assert svc.db.run_select("SELECT COUNT(*) FROM jobs")[1] == [(1,)]
    assert svc.db.run_select("SELECT times_posted FROM jobs")[1] == [(2,)]


def test_exact_repost_skips_llm(make_service):
    svc = make_service([GOOD_JSON])
    svc.ingest("Zephyr Labs hiring Data Analyst Intern in Pune 15-20k. Apply https://forms.gle/abc123")
    n_calls = len(svc.llm.calls)
    r = svc.ingest("zephyr labs hiring  data analyst intern in pune 15-20k. apply https://forms.gle/abc123")
    assert r["status"] == "exact_duplicate" and len(svc.llm.calls) == n_calls


def test_different_company_not_duplicate(make_service):
    other = GOOD_JSON.replace("Zephyr Labs", "Totally Different Corp")
    svc = make_service([GOOD_JSON, other])
    svc.ingest("Zephyr Labs hiring Data Analyst Intern in Pune 15-20k. Apply https://forms.gle/abc123")
    r = svc.ingest("Totally Different Corp hiring Data Analyst Intern in Pune 15-20k. Apply forms.gle/abc123")
    assert r["status"] == "new"


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder()
    a, b = e.embed(["python developer pune"]), e.embed(["python developer pune"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a[0]) - 1) < 1e-5
