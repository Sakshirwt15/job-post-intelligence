import pytest

from app.db import Database
from app.embeddings import HashEmbedder
from app.llm import LLMResult
from app.service import JobIntelligence
from app.vectorstore import InMemoryVectorStore


class FakeLLM:
    """Pre-programmed responses; queue khatam hone par last response repeat hota hai."""

    provider, model = "fake", "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, *, system=None, purpose="generic", prompt_version=None, json_mode=False, temperature=0.0):
        self.calls.append({"prompt": prompt, "purpose": purpose})
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return LLMResult(text=text, prompt_tokens=10, completion_tokens=5, latency_ms=1.0, model="fake", provider="fake")


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def make_service(db):
    def _make(responses, threshold=0.88):
        return JobIntelligence(db, FakeLLM(responses), HashEmbedder(), InMemoryVectorStore(), dedup_threshold=threshold)
    return _make


GOOD_JSON = ('{"company":"Zephyr Labs","role":"Data Analyst Intern","employment_type":"internship","batch":[2025,2026],'
             '"stipend_min":"15k","stipend_max":20000,"location":"Pune","is_remote":false,'
             '"skills":["SQL","Excel"],"apply_link":"https://forms.gle/abc123","apply_email":null}')
