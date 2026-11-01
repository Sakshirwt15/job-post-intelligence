from app.monitoring import summarize
from eval.metrics import aggregate, field_scores, results_match

GOLD = {"company": "Acme Labs", "role": "AI/ML Intern", "employment_type": "internship", "batch": [2025, 2026],
        "stipend_min": 20000, "stipend_max": 30000, "location": "Pune", "is_remote": False,
        "skills": ["Python", "SQL"], "apply_link": "https://x.io/a", "apply_email": None}


def test_perfect_prediction_scores_one():
    s = field_scores({**GOLD, "role": "AI ML Intern", "location": "pune"}, GOLD)
    assert all(v == 1.0 for v in s.values())


def test_wrong_fields_and_failures():
    s = field_scores({**GOLD, "stipend_min": 2000, "skills": ["Python"]}, GOLD)
    assert s["stipend_min"] == 0.0 and 0 < s["skills"] < 1
    assert aggregate([field_scores(None, GOLD)])["overall"] == 0.0


def test_results_match_exact_and_projection():
    assert results_match([(3,)], [(3,)])
    assert results_match([("a",), ("b",)], [("b",), ("a",)])
    assert results_match([("Acme",)], [(1, "Acme")])  # extra column ok
    assert not results_match([(3,)], [(4,)])
    assert results_match([(2.0,)], [(2.004,)])


def test_monitoring_summary():
    logs = [{"ts": "2026-11-01T10:00:00", "purpose": "extraction", "model": "m", "prompt_version": "v1",
             "latency_ms": 100, "prompt_tokens": 10, "completion_tokens": 5, "success": 1, "error": None},
            {"ts": "2026-11-01T10:01:00", "purpose": "text2sql", "model": "m", "prompt_version": "v1",
             "latency_ms": 300, "prompt_tokens": 20, "completion_tokens": 0, "success": 0, "error": "HTTP 429"}]
    s = summarize(logs)
    assert s["total_calls"] == 2 and s["error_rate"] == 0.5 and s["avg_latency_ms"] == 200.0
    assert s["recent_errors"][0]["error"] == "HTTP 429"
    assert summarize([]) == {"total_calls": 0}
