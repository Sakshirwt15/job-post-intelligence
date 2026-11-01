"""Pydantic models - LLM ka output yahin validate hota hai. Fail hone par extraction retry hota hai."""
import re
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.textutil import job_to_text

EmploymentType = Literal["internship", "full-time", "contract", "unknown"]
_PLACEHOLDERS = {"", "null", "none", "n/a", "na", "unknown", "not mentioned", "not specified", "-"}
MIN_BATCH, MAX_BATCH = 2015, 2035


def _parse_money(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(round(v))
    s = str(v).strip().lower()
    if s in _PLACEHOLDERS or s in {"unpaid", "performance based", "competitive"}:
        return None
    s = re.sub(r"(₹|rs\.?|inr|/-|per month|/month|pm|/mo)", "", s).replace(",", "").strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k|l|lac|lakh|lpa)?", s)
    if not m:
        raise ValueError(f"Cannot parse stipend value: {v!r} (use a plain monthly INR number)")
    num, unit = float(m.group(1)), m.group(2)
    if unit == "k":
        num *= 1_000
    elif unit in {"l", "lac", "lakh"}:
        num *= 100_000
    elif unit == "lpa":
        num = num * 100_000 / 12
    return int(round(num))


class JobPost(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    company: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=160)
    employment_type: EmploymentType = "unknown"
    batch: list[int] = Field(default_factory=list)
    stipend_min: Optional[int] = Field(default=None, ge=0, le=10_000_000)  # INR / month
    stipend_max: Optional[int] = Field(default=None, ge=0, le=10_000_000)  # INR / month
    location: Optional[str] = Field(default=None, max_length=80)
    is_remote: bool = False
    skills: list[str] = Field(default_factory=list)
    apply_link: Optional[str] = None
    apply_email: Optional[str] = None

    # ---------- before-validators: LLM ki messy output ko tolerate karo ----------
    @model_validator(mode="before")
    @classmethod
    def _remote_location(cls, data):
        if isinstance(data, dict):
            loc = data.get("location")
            if isinstance(loc, str) and loc.strip().lower() in {"remote", "wfh", "work from home"}:
                data = {**data, "location": None, "is_remote": True}
        return data

    @field_validator("company", "role", mode="before")
    @classmethod
    def _no_placeholder(cls, v):
        if v is None or (isinstance(v, str) and v.strip().lower() in _PLACEHOLDERS):
            raise ValueError("must be a real value, not null/unknown")
        return v

    @field_validator("employment_type", mode="before")
    @classmethod
    def _emp_type(cls, v):
        s = str(v or "unknown").strip().lower().replace("_", "-")
        if s in {"intern", "internship", "internships", "stipend"}:
            return "internship"
        if s in {"full-time", "full time", "fulltime", "fte", "permanent", "job"}:
            return "full-time"
        if s in {"contract", "freelance", "part-time", "part time"}:
            return "contract"
        return "unknown"

    @field_validator("stipend_min", "stipend_max", mode="before")
    @classmethod
    def _money(cls, v):
        return _parse_money(v)

    @field_validator("batch", mode="before")
    @classmethod
    def _batch_before(cls, v):
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        years: list[int] = []
        for it in items:
             years += [int(y) for y in re.findall(r"(?<!\d)(\d{4})(?!\d)", str(it))]
        return sorted(set(years))

    @field_validator("batch")
    @classmethod
    def _batch_range(cls, v):
        bad = [y for y in v if not MIN_BATCH <= y <= MAX_BATCH]
        if bad:
            raise ValueError(f"batch years out of range {MIN_BATCH}-{MAX_BATCH}: {bad}")
        return v

    @field_validator("location", mode="before")
    @classmethod
    def _loc(cls, v):
        if v is None or str(v).strip().lower() in _PLACEHOLDERS:
            return None
        return str(v).strip()

    @field_validator("skills", mode="before")
    @classmethod
    def _skills_before(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = re.split(r"[,/|;]", v)
        return [str(x).strip() for x in v]

    @field_validator("skills")
    @classmethod
    def _skills_clean(cls, v):
        seen, out = set(), []
        for s in v:
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out[:30]

    @field_validator("apply_link", mode="before")
    @classmethod
    def _link(cls, v):
        if v is None or str(v).strip().lower() in _PLACEHOLDERS:
            return None
        s = str(v).strip().strip("<>()[]\"'.,")
        if not re.match(r"^https?://", s, flags=re.IGNORECASE):
            if re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", s, flags=re.IGNORECASE):
                s = "https://" + s
            else:
                raise ValueError(f"apply_link is not a valid URL: {v!r}")
        u = urlparse(s)
        if u.scheme not in {"http", "https"} or "." not in u.netloc:
            raise ValueError(f"apply_link is not a valid URL: {v!r}")
        return s

    @field_validator("apply_email", mode="before")
    @classmethod
    def _email(cls, v):
        if v is None or str(v).strip().lower() in _PLACEHOLDERS:
            return None
        s = str(v).strip().strip("<>()[]\"'.,")
        if not re.fullmatch(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", s):
            raise ValueError(f"apply_email is not a valid email: {v!r}")
        return s

    @model_validator(mode="after")
    def _stipend_order(self):
        if self.stipend_min is not None and self.stipend_max is not None and self.stipend_min > self.stipend_max:
            raise ValueError("stipend_min cannot be greater than stipend_max")
        return self

    def to_embedding_text(self) -> str:
        return job_to_text(self.model_dump())


# ---------- API request models ----------
class IngestRequest(BaseModel):
    raw_text: str = Field(min_length=20)
    prompt_version: Optional[str] = None


class BulkIngestRequest(BaseModel):
    posts: list[str]


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(default=5, ge=1, le=20)


class SQLRequest(BaseModel):
    question: str = Field(min_length=3)
