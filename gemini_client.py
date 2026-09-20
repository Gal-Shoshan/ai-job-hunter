"""Gemini-backed assessment of job listings against the user's resume.

    from job_search import search_jobs
    from gemini_client import analyze_jobs

    jobs = search_jobs(SITES, ["backend engineer"], limit=10)
    results = analyze_jobs("resume.pdf", "backend development", jobs)

The answer template on disk is the single source of truth for the output
shape: a Gemini response schema is derived from it, the reply is parsed into
a dict, and the dict is verified to carry exactly the template's keys with
no empty values. Requires GEMINI_API_KEY in the environment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from google import genai
from google.genai import types

from job_search import JobListing

LOG = logging.getLogger(__name__)

# `genai.Client().models.list()` prints what your key can actually reach.
#
# Flash-Lite is the cheaper, higher-throughput half of the family, meant for
# extract-and-classify work like this one: read a listing, fill in a fixed
# schema. Its per-minute and per-day allowances are considerably larger than
# the full Flash models', which is what makes a whole cycle fit in a day.
#
# Retired models answer 404 "no longer available to new users" on every
# call, which shows up here as one failed assessment per listing and an
# empty result set. If that happens, this line is what to change.
MODEL = "gemini-3.5-flash-lite"

TEMPLATE_PATH = Path(os.environ.get("JOB_TEMPLATE", "llm_format_answer.json"))

# Inline attachment cap for the Gemini API. Larger files need the Files API.
MAX_INLINE_BYTES = 20 * 1024 * 1024

# A free-tier key allows only a set number of generate_content calls per
# minute per model, and answers 429 RESOURCE_EXHAUSTED above that. Calls are
# spaced to use that allowance fully without crossing it. The real figure
# for this project is on https://aistudio.google.com/rate-limit; set this to
# match it, raise it once the key is on a paid tier, or use 0 for no pacing.
REQUESTS_PER_MINUTE = 10.0

# 429 (over quota) and 5xx (model briefly overloaded) are worth waiting out.
# They say nothing about the listing, so failing it outright throws away a
# job for a reason that would have cleared in half a minute.
TRANSPORT_ATTEMPTS = 4
MAX_RETRY_WAIT = 90.0
RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_STATUSES = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL")

_last_call_at = 0.0

SYSTEM_INSTRUCTION = (
    "You assess how well a single job listing matches a candidate's resume. "
    "Base every judgement only on the resume and the listing provided; never "
    "invent experience the candidate does not demonstrate. "
    "Fill in every field of the required schema — no field may be left null, "
    "empty, or set to a placeholder. When the listing genuinely does not state "
    "something, write \"not specified\" rather than leaving the field blank. "
    "Where a field says an estimate is always required, refusing to answer is "
    "not an option: give your best figure from the role, the seniority and "
    "the local market, and mark it as estimated. "
    "Return the JSON object only: no commentary and no markdown fences."
)


class GeminiError(RuntimeError):
    """Raised when the model call fails or returns unusable output."""


class GeminiConfigError(GeminiError):
    """Raised when the setup itself is wrong: no key, no template, no CV.

    These affect every job equally, so analyze_jobs re-raises them instead
    of logging the same warning once per listing.
    """


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def analyze_job(
    resume_path: str | Path,
    job_field: str,
    job: JobListing | dict[str, Any],
    *,
    home_location: str | None = None,
    attempts: int = 2,
) -> dict[str, Any]:
    """Score one listing against the resume.

    Returns a dict with exactly the keys of the answer template, in the
    template's own order, every field populated. Raises GeminiError if the
    model cannot produce that after `attempts` tries.

    home_location gives the model an origin for the commute estimate, e.g.
    "Rishpon, Israel". Without it that field is guesswork; even with it the
    figure is the model's impression, not a routed drive time.
    """
    template = _load_template()
    resume_part = _resume_part(Path(resume_path))

    home_line = (
        f"I live in {home_location}, which is where any commute is measured from.\n"
        if home_location else
        "My home location is not given, so any commute estimate must be 0.\n"
    )

    prompt = f"""I am a computer science student looking for a position in the field of: {job_field}.
{home_line}
The attached PDF is my CV. Review it, then review this job listing:

{json.dumps(_to_dict(job), indent=2, ensure_ascii=False)}

Return a JSON object with exactly these keys, every one filled in:

{json.dumps(template, indent=2, ensure_ascii=False)}"""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=_response_schema(),
        temperature=0.2,
    )

    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = _call_model([resume_part, prompt], config)
            parsed = _parse_json_response(response)
            _validate_against_template(parsed, template)
            _require_estimates(parsed, template,
                               last_attempt=attempt >= max(1, attempts))
            return _ordered_like(parsed, template)
        except GeminiConfigError:
            raise                      # retrying will not fix the setup
        except GeminiError as exc:
            last_error = exc
            LOG.warning("Attempt %d/%d rejected: %s", attempt, attempts, exc)

    raise GeminiError(f"No valid response after {attempts} attempts: {last_error}")


def _call_model(contents: list[Any], config: Any) -> Any:
    """generate_content, paced for the quota and retried when the API asks.

    Quota and overload errors are not verdicts on the listing, so they are
    waited out here instead of bubbling up to analyze_jobs, which would drop
    the job and only look at it again on the next cycle.
    """
    last_error: Exception | None = None

    for attempt in range(1, max(1, TRANSPORT_ATTEMPTS) + 1):
        _throttle()
        try:
            return _client().models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except Exception as exc:
            if not _is_retryable(exc) or attempt == TRANSPORT_ATTEMPTS:
                raise
            last_error = exc
            wait = _retry_after(exc, attempt)
            LOG.info("%s; waiting %.0fs then retrying (%d/%d).",
                     _status_name(exc), wait, attempt, TRANSPORT_ATTEMPTS)
            time.sleep(wait)

    raise last_error   # unreachable: the final attempt re-raises


def _throttle() -> None:
    """Hold each call at least one quota slot after the previous one."""
    global _last_call_at
    if REQUESTS_PER_MINUTE <= 0:
        return
    gap = 60.0 / REQUESTS_PER_MINUTE
    wait = gap - (time.monotonic() - _last_call_at)
    if wait > 0:
        LOG.debug("Pacing: waiting %.1fs before the next call.", wait)
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (GeminiConfigError, GeminiError)):
        return False           # our own validation errors, not the transport
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in RETRYABLE_CODES
    text = str(exc)
    return any(status in text for status in _RETRYABLE_STATUSES)


def _retry_after(exc: Exception, attempt: int) -> float:
    """How long to wait: the API usually states it, otherwise back off."""
    text = str(exc)
    for pattern in (r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'",
                    r"retry in (\d+(?:\.\d+)?)s"):
        match = re.search(pattern, text)
        if match:
            return min(float(match.group(1)) + 1.0, MAX_RETRY_WAIT)
    return min(5.0 * 2 ** attempt, MAX_RETRY_WAIT)


def _status_name(exc: Exception) -> str:
    for status in _RETRYABLE_STATUSES:
        if status in str(exc):
            return status
    return f"{getattr(exc, 'code', type(exc).__name__)}"


def analyze_jobs(
    resume_path: str | Path,
    job_field: str,
    jobs: Sequence[JobListing | dict[str, Any]],
    *,
    home_location: str | None = None,
    skip_failures: bool = True,
) -> list[dict[str, Any]]:
    """Assess a batch of listings, attaching each result to its source job."""
    results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        label = _to_dict(job).get("title", f"job #{index}")
        try:
            assessment = analyze_job(
                resume_path, job_field, job, home_location=home_location
            )
        except GeminiConfigError:
            # Broken setup, not a bad listing: stop rather than repeat this
            # warning for every job in the batch.
            raise
        except Exception as exc:
            LOG.warning("Assessment failed for %s: %s", label, exc)
            if not skip_failures:
                raise
            continue
        results.append({"job": _to_dict(job), "assessment": assessment})
        LOG.info("Assessed %d/%d: %s", index, len(jobs), label)
    return results


# --------------------------------------------------------------------------- #
# Template -> response schema
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _response_schema() -> types.Schema:
    """Derive a Gemini response schema from the answer template.

    Template conventions, all of which keep the file readable as a sample
    answer while giving the model a typed, described field:

      "free text"           STRING; the text becomes the field description
      "A | B | C"           STRING constrained to that set of values
      "<int> what to put"   INTEGER with the rest as its description
      "<number> ..."        NUMBER          "<bool> ..."   BOOLEAN
      0 / 0.0 / true        typed literally, with no description
      ["an item"]           ARRAY of the first element's type
      {...}                 nested OBJECT, every key required
    """
    return _schema_for(_load_template())


_TYPE_TAG = re.compile(r"^<(int|integer|number|float|bool|boolean|str|string)>\s*")

_TAG_TYPES = {
    "int": types.Type.INTEGER, "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER, "float": types.Type.NUMBER,
    "bool": types.Type.BOOLEAN, "boolean": types.Type.BOOLEAN,
    "str": types.Type.STRING, "string": types.Type.STRING,
}


def _schema_for(value: Any, path: str = "") -> types.Schema:
    where = path or "root"

    if isinstance(value, dict):
        keys = list(value.keys())
        return types.Schema(
            type=types.Type.OBJECT,
            properties={
                k: _schema_for(v, f"{path}.{k}" if path else k)
                for k, v in value.items()
            },
            required=keys,
            property_ordering=keys,
        )
    if isinstance(value, list):
        return types.Schema(
            type=types.Type.ARRAY,
            items=_schema_for(value[0] if value else "", f"{where}[]"),
        )
    if isinstance(value, bool):  # must precede int; bool is an int subclass
        return types.Schema(type=types.Type.BOOLEAN)
    if isinstance(value, int):
        return types.Schema(type=types.Type.INTEGER)
    if isinstance(value, float):
        return types.Schema(type=types.Type.NUMBER)

    if isinstance(value, str):
        tag = _TYPE_TAG.match(value)
        if tag:
            return types.Schema(
                type=_TAG_TYPES[tag.group(1)],
                description=value[tag.end():].strip() or None,
            )
        options = _enum_options(value)
        if options:
            return types.Schema(
                type=types.Type.STRING,
                enum=options,
                description="exactly one of: " + ", ".join(options),
            )
        # A descriptive placeholder ("one-line summary") becomes the
        # field description, which is what steers the model's content.
        return types.Schema(type=types.Type.STRING, description=value or None)

    LOG.warning(
        "Template field %r is null, so it carries no type or guidance. "
        "Replace it with a description, a literal like 0, or a '<int> ...' tag.",
        where,
    )
    return types.Schema(type=types.Type.STRING)


def _enum_options(value: str) -> list[str] | None:
    """Read 'A | B | C' as a closed set of allowed values."""
    if "|" not in value:
        return None
    options = [part.strip() for part in value.split("|")]
    return options if len(options) >= 2 and all(options) else None


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #

def _parse_json_response(response: Any) -> Any:
    """Turn the model's reply text into Python data."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, (dict, list)):
        return parsed  # SDK already decoded it

    text = getattr(response, "text", None)
    if not text:
        feedback = getattr(response, "prompt_feedback", None)
        raise GeminiError(f"Model returned no text (feedback: {feedback}).")

    cleaned = text.strip()
    if cleaned.startswith("```"):  # belt and braces; schema mode shouldn't fence
        cleaned = cleaned.split("```")[1]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GeminiError(
            f"Model did not return valid JSON: {exc}\n--- raw ---\n{text[:500]}"
        ) from exc


def _validate_against_template(parsed: Any, template: Any, path: str = "") -> None:
    """Assert the reply matches the template's keys and has no empty fields."""
    where = path or "root"

    if isinstance(template, dict):
        if not isinstance(parsed, dict):
            raise GeminiError(f"{where}: expected an object, got {type(parsed).__name__}.")
        missing = [k for k in template if k not in parsed]
        extra = [k for k in parsed if k not in template]
        if missing or extra:
            raise GeminiError(
                f"{where}: key mismatch (missing: {missing or 'none'}, "
                f"unexpected: {extra or 'none'})."
            )
        for key, sub in template.items():
            _validate_against_template(parsed[key], sub, f"{path}.{key}" if path else key)
        return

    if isinstance(template, list):
        if not isinstance(parsed, list):
            raise GeminiError(f"{where}: expected an array, got {type(parsed).__name__}.")
        if template:
            for i, item in enumerate(parsed):
                _validate_against_template(item, template[0], f"{where}[{i}]")
        return

    if parsed is None or (isinstance(parsed, str) and not parsed.strip()):
        raise GeminiError(f"{where}: field came back empty.")


# A template field whose description says so must come back as a real
# number. The template stays the single source of truth: mark a field by
# writing "never 0" in its description rather than naming it here.
_MUST_ESTIMATE = re.compile(r"never\s*0", re.I)


def _require_estimates(parsed: Any, template: Any,
                       *, last_attempt: bool, path: str = "") -> None:
    """Reject zeros in fields the template insists on an estimate for.

    Raising sends the listing back for another attempt. On the final try the
    value is kept and logged instead: a missing salary estimate is a poorer
    assessment, not a reason to throw the whole listing away.
    """
    if isinstance(template, dict) and isinstance(parsed, dict):
        for key, sub_template in template.items():
            if key in parsed:
                _require_estimates(parsed[key], sub_template,
                                   last_attempt=last_attempt,
                                   path=f"{path}.{key}" if path else key)
        return

    if not (isinstance(template, str) and _MUST_ESTIMATE.search(template)):
        return
    if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
        return
    if parsed != 0:
        return

    if last_attempt:
        LOG.warning("%s came back 0 although the template requires an "
                    "estimate; keeping it.", path or "field")
        return
    raise GeminiError(f"{path or 'field'}: an estimate is required, got 0.")


def _ordered_like(parsed: Any, template: Any) -> Any:
    """Rebuild the reply in the template's key order."""
    if isinstance(template, dict) and isinstance(parsed, dict):
        return {k: _ordered_like(parsed[k], v) for k, v in template.items()}
    if isinstance(template, list) and isinstance(parsed, list) and template:
        return [_ordered_like(item, template[0]) for item in parsed]
    return parsed


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _client() -> genai.Client:
    """Built on first use, so importing this module never needs a key."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise GeminiConfigError(
            "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com "
            "and export it before running."
        )
    return genai.Client()


@lru_cache(maxsize=1)
def _load_template() -> dict[str, Any]:
    """Read the answer-format template from disk (it is data, not a module)."""
    if not TEMPLATE_PATH.is_file():
        raise GeminiConfigError(f"Answer template not found at {TEMPLATE_PATH}.")
    try:
        template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GeminiConfigError(f"{TEMPLATE_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(template, dict):
        raise GeminiConfigError(f"{TEMPLATE_PATH} must contain a JSON object at the top level.")
    return template


@lru_cache(maxsize=4)
def _resume_bytes(path_str: str, mtime: float) -> bytes:
    """Cached on (path, mtime) so a long run reads the PDF only once."""
    path = Path(path_str)
    data = path.read_bytes()
    if len(data) > MAX_INLINE_BYTES:
        raise GeminiError(
            f"{path} is {len(data) / 1e6:.1f} MB; inline attachments are capped "
            f"at {MAX_INLINE_BYTES / 1e6:.0f} MB. Use client.files.upload() instead."
        )
    if not data.startswith(b"%PDF"):
        raise GeminiConfigError(f"{path} does not look like a PDF.")
    return data


def _resume_part(path: Path) -> types.Part:
    if not path.is_file():
        raise GeminiConfigError(f"Resume not found at {path}.")
    data = _resume_bytes(str(path), path.stat().st_mtime)
    return types.Part.from_bytes(data=data, mime_type="application/pdf")


def _to_dict(job: JobListing | dict[str, Any]) -> dict[str, Any]:
    """Send the model structured fields rather than a dataclass repr."""
    if isinstance(job, dict):
        return job
    if is_dataclass(job):
        return {k: v for k, v in asdict(job).items() if v is not None}
    return {"listing": str(job)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from job_search import SiteConfig, search_jobs

    sites = [
        SiteConfig(
            name="Remotive",
            search_url="https://remotive.com/remote-jobs?search={query}",
        ),
    ]

    listings = search_jobs(sites, ["python developer"], limit=3)
    for entry in analyze_jobs(
        "resume.pdf", "backend development", listings, home_location="Rishpon, Israel"
    ):
        print(entry["job"].get("title"))
        print(json.dumps(entry["assessment"], indent=2, ensure_ascii=False))
        print()