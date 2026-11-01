import pytest

from app.extraction import ExtractionError, extract_job
from tests.conftest import GOOD_JSON, FakeLLM


def test_extracts_and_validates():
    r = extract_job(FakeLLM([GOOD_JSON]), "raw post text", "extract_v2")
    assert r.attempts == 1
    assert r.job.stipend_min == 15000 and r.job.batch == [2025, 2026]


def test_retries_on_bad_json_then_succeeds():
    llm = FakeLLM(["not json at all", GOOD_JSON])
    r = extract_job(llm, "raw post text", "extract_v2")
    assert r.attempts == 2 and len(r.errors) == 1
    assert "rejected by the validator" in llm.calls[1]["prompt"]


def test_retries_on_validation_error():
    bad = GOOD_JSON.replace('"https://forms.gle/abc123"', '"no link here"')
    r = extract_job(FakeLLM([bad, GOOD_JSON]), "raw post text", "extract_v2")
    assert r.attempts == 2


def test_fails_after_max_attempts():
    with pytest.raises(ExtractionError):
        extract_job(FakeLLM(["{}"]), "raw post text", "extract_v2", max_attempts=3)


def test_json_in_code_fence_is_accepted():
    r = extract_job(FakeLLM([f"```json\n{GOOD_JSON}\n```"]), "raw post text", "extract_v1")
    assert r.job.company == "Zephyr Labs"
