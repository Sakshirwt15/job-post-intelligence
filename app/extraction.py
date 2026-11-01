"""Raw post -> LLM -> JSON -> Pydantic validation (galat output par feedback ke saath retry)."""
import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from app import config
from app.llm import LLMClient
from app.prompts import load_prompt
from app.schemas import JobPost
from app.textutil import parse_json_loose

SYSTEM = "You are a precise information-extraction engine. Reply with ONLY one valid JSON object, no prose."


class ExtractionError(RuntimeError):
    def __init__(self, message, errors=None, usage=None):
        super().__init__(message)
        self.errors = errors or []
        self.usage = usage or {}


@dataclass
class ExtractionResult:
    job: JobPost
    attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


def _format_validation_error(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in err['loc']) or 'root'}: {err['msg']}" for err in e.errors())


def extract_job(llm: LLMClient, raw_text: str, prompt_version: str | None = None,
                max_attempts: int | None = None) -> ExtractionResult:
    version = prompt_version or config.EXTRACT_PROMPT_VERSION
    max_attempts = max_attempts or config.MAX_EXTRACT_ATTEMPTS
    base = load_prompt(version).replace("{{POST}}", raw_text.strip())
    prompt, errors = base, []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}

    for attempt in range(1, max_attempts + 1):
        out = llm.generate(prompt, system=SYSTEM, purpose="extraction", prompt_version=version, json_mode=True)
        usage["prompt_tokens"] += out.prompt_tokens
        usage["completion_tokens"] += out.completion_tokens
        usage["latency_ms"] += out.latency_ms
        try:
            job = JobPost.model_validate(parse_json_loose(out.text))
            return ExtractionResult(job, attempt, errors=errors, **usage)
        except ValidationError as e:
            problem = _format_validation_error(e)
        except (ValueError, json.JSONDecodeError) as e:
            problem = f"Invalid JSON: {e}"
        errors.append(problem)
        prompt = (base + "\n\nYour previous answer was rejected by the validator:\n"
                  f"{out.text[:1500]}\n\nProblems: {problem}\n"
                  "Fix these problems and return the corrected JSON object only.")

    raise ExtractionError(f"Extraction failed after {max_attempts} attempts: {errors[-1]}", errors, usage)
