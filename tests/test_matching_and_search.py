import json

from app.matching import combine_score, keyword_overlap, scale_similarity
from tests.conftest import GOOD_JSON

RESUME = "Final year student. Skills: Python, SQL, Excel, scikit-learn, ML projects. Built Power BI dashboards."


def test_keyword_overlap_with_aliases():
    score, matched, missing = keyword_overlap(["Python", "Machine Learning", "Docker"], RESUME)
    assert matched == ["Python", "Machine Learning"] and missing == ["Docker"]
    assert round(score, 2) == 0.67


def test_keyword_overlap_no_substring_false_positive():
    assert keyword_overlap(["Java"], "I know JavaScript")[1] == []
    assert keyword_overlap(["C++"], "worked in c++ and rust")[1] == ["C++"]


def test_no_skills_returns_none_and_embedding_only_score():
    assert keyword_overlap([], RESUME)[0] is None
    assert combine_score(0.8, None) == 0.8


def test_scale_similarity_bounds():
    assert scale_similarity(0.0) == 0.0 and scale_similarity(1.0) == 1.0


def test_resume_ranking_prefers_skill_match(make_service):
    other = GOOD_JSON.replace("Zephyr Labs", "Docker Corp").replace('"SQL","Excel"', '"Docker","Kubernetes"')
    svc = make_service([GOOD_JSON, other])
    svc.ingest("Zephyr Labs hiring Data Analyst Intern in Pune 15-20k. Apply https://forms.gle/abc123")
    svc.ingest("Docker Corp hiring DevOps style intern in Pune. Apply https://forms.gle/zzz999 Docker Kubernetes")
    ranked = svc.match_resume(RESUME)
    assert ranked[0]["company"] == "Zephyr Labs" and ranked[0]["score"] > ranked[1]["score"]


def test_search_applies_parsed_filters_and_returns_sources(make_service):
    parse = json.dumps({"english_query": "data analyst internship in Pune", "location": "Pune",
                        "min_stipend": 10000, "employment_type": "internship", "skills": []})
    svc = make_service([GOOD_JSON])
    svc.ingest("Zephyr Labs hiring Data Analyst Intern in Pune 15-20k. Apply https://forms.gle/abc123")
    svc.llm.responses = [parse, "Zephyr Labs ka Data Analyst Intern role [1] Pune me hai."]
    out = svc.search("pune me 10k se upar data internship")
    assert out["parsed"]["location"] == "Pune" and not out["filters_relaxed"]
    assert [s["id"] for s in out["sources"]] == [1] and "[1]" in out["answer"]


def test_search_survives_llm_failure(make_service):
    svc = make_service([GOOD_JSON])
    svc.ingest("Zephyr Labs hiring Data Analyst Intern in Pune 15-20k. Apply https://forms.gle/abc123")
    svc.llm.responses = ["garbage not json"]
    assert svc.search("data analyst")["sources"]
