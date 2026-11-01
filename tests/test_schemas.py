import pytest
from pydantic import ValidationError

from app.schemas import JobPost

BASE = {"company": "Acme", "role": "ML Intern"}


def test_money_and_batch_coercion():
    j = JobPost(**BASE, stipend_min="25k", stipend_max="₹30,000", batch="2025/2026", skills="Python, SQL, python")
    assert (j.stipend_min, j.stipend_max) == (25000, 30000)
    assert j.batch == [2025, 2026]
    assert j.skills == ["Python", "SQL"]


def test_lpa_converted_to_monthly():
    assert JobPost(**BASE, stipend_min="6 LPA").stipend_min == 50000


def test_invalid_link_rejected():
    with pytest.raises(ValidationError):
        JobPost(**BASE, apply_link="click the link in bio")


def test_link_without_scheme_gets_https():
    assert JobPost(**BASE, apply_link="kiteflow.io/careers/x").apply_link == "https://kiteflow.io/careers/x"


def test_batch_out_of_range_rejected():
    with pytest.raises(ValidationError):
        JobPost(**BASE, batch=[1999])


def test_min_greater_than_max_rejected():
    with pytest.raises(ValidationError):
        JobPost(**BASE, stipend_min=40000, stipend_max=20000)


def test_unparseable_stipend_rejected():
    with pytest.raises(ValidationError):
        JobPost(**BASE, stipend_min="good stipend")


def test_remote_location_sets_flag():
    j = JobPost(**BASE, location="Remote")
    assert j.location is None and j.is_remote is True


def test_placeholder_company_rejected():
    with pytest.raises(ValidationError):
        JobPost(company="N/A", role="Dev")
