"""Gemini-backed assessment of job listings against the user's resume.

Each listing is sent to the model together with the resume as an inline
PDF, and comes back as a JSON object whose shape is dictated by an answer
template on disk. That template is the single source of truth: a Gemini
response schema is derived from it, the reply is parsed against it, and the
result is verified to carry exactly its keys with no empty values.

Example:
    from gemini_client import analyze_jobs

    results = analyze_jobs("resume.pdf", "backend development", listings)

Environment:
    GEMINI_API_KEY or GOOGLE_API_KEY: API credentials. Required.
    JOB_TEMPLATE: path to the answer template. Defaults to
        ``llm_format_answer.json``.

Note:
    Calls are paced to REQUESTS_PER_MINUTE and retried on quota and
    overload errors, which say nothing about the listing itself.
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

MODEL = "gemini-3.5-flash-lite"

TEMPLATE_PATH = Path(os.environ.get("JOB_TEMPLATE", "llm_format_answer.json"))

MAX_INLINE_BYTES = 20 * 1024 * 1024

REQUESTS_PER_MINUTE = 10.0

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
    """Raised when a model call fails or returns unusable output."""


class GeminiConfigError(GeminiError):
    """Raised when the setup itself is wrong: no key, no template, no CV.

    These affect every listing equally, so ``analyze_jobs`` re-raises them
    rather than logging the same warning once per listing.
    """


def analyze_job(
    resume_path: str | Path,
    job_field: str,
    job: JobListing | dict[str, Any],
    *,
    home_location: str | None = None,
    attempts: int = 2,
) -> dict[str, Any]:
    """Score a single listing against the resume.

    Args:
        resume_path: Path to the resume PDF, sent inline with the prompt.
        job_field: The kind of role being sought, in plain words, used to
            orient the model.
        job: A JobListing or an equivalent mapping.
        home_location: Origin for the commute estimate, e.g.
            ``"Shefayim, Israel"``. Without it the model is told to return 0.
            The figure is the model's impression, not a routed drive time.
        attempts: How many times to ask before giving up. Defaults to 2.

    Returns:
        A dict carrying exactly the answer template's keys, in the template's
        order, every field populated.

    Raises:
        GeminiConfigError: The key, template or resume is missing or unusable.
        GeminiError: No valid response after ``attempts`` tries.
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
            raise
        except GeminiError as exc:
            last_error = exc
            LOG.warning("Attempt %d/%d rejected: %s", attempt, attempts, exc)

    raise GeminiError(f"No valid response after {attempts} attempts: {last_error}")


def _call_model(contents: list[Any], config: Any) -> Any:
    """Call generate_content, paced for the quota and retried when asked.

    Quota and overload errors are not verdicts on the listing, so they are
    waited out here instead of reaching ``analyze_jobs``, which would drop the
    job and only look at it again on the next cycle.

    Args:
        contents: Prompt parts to send, including the resume.
        config: Generation config, carrying the schema and system instruction.

    Returns:
        The raw SDK response object.

    Raises:
        Exception: Whatever the SDK raised, once the error is not retryable or
            the attempts are exhausted.
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

    raise last_error


def _throttle() -> None:
    """Hold each call at least one quota slot after the previous one.

    Sleeps just long enough to keep the call rate at REQUESTS_PER_MINUTE.
    Pacing is disabled when that constant is zero or less.
    """
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
    """Report whether an error is worth waiting out.

    Args:
        exc: The exception raised by the SDK.

    Returns:
        True for quota and transient server errors, which clear on their own.
        False for this module's own validation errors and for client errors,
        which repeating would not fix.
    """
    if isinstance(exc, (GeminiConfigError, GeminiError)):
        return False
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in RETRYABLE_CODES
    text = str(exc)
    return any(status in text for status in _RETRYABLE_STATUSES)


def _retry_after(exc: Exception, attempt: int) -> float:
    """Decide how long to wait before repeating a call.

    Args:
        exc: The retryable exception, which often states its own delay.
        attempt: Which attempt has just failed, counting from one.

    Returns:
        Seconds to wait: the API's stated ``retryDelay`` when present, else an
        exponential back-off, capped at MAX_RETRY_WAIT.
    """
    text = str(exc)
    for pattern in (r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'",
                    r"retry in (\d+(?:\.\d+)?)s"):
        match = re.search(pattern, text)
        if match:
            return min(float(match.group(1)) + 1.0, MAX_RETRY_WAIT)
    return min(5.0 * 2 ** attempt, MAX_RETRY_WAIT)


def _status_name(exc: Exception) -> str:
    """Summarise an error for a single log line.

    Args:
        exc: The exception raised by the SDK.

    Returns:
        The API status name when recognisable, else the status code or the
        exception's type name.
    """
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
    """Assess a batch of listings, pairing each result with its listing.

    Args:
        resume_path: Path to the resume PDF.
        job_field: The kind of role being sought, in plain words.
        jobs: Listings to assess.
        home_location: Origin for the commute estimate, or None.
        skip_failures: Log and continue past a listing that could not be
            assessed rather than aborting the batch. Defaults to True.

    Returns:
        One ``{"job": ..., "assessment": ...}`` entry per listing that was
        assessed successfully, in input order. Listings that failed are
        omitted, so the result may be shorter than ``jobs``.

    Raises:
        GeminiConfigError: The setup is broken, which would fail every
            listing alike.
        GeminiError: A listing could not be assessed and ``skip_failures``
            is False.
    """
    results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        label = _to_dict(job).get("title", f"job #{index}")
        try:
            assessment = analyze_job(
                resume_path, job_field, job, home_location=home_location
            )
        except GeminiConfigError:
            raise
        except Exception as exc:
            LOG.warning("Assessment failed for %s: %s", label, exc)
            if not skip_failures:
                raise
            continue
        results.append({"job": _to_dict(job), "assessment": assessment})
        LOG.info("Assessed %d/%d: %s", index, len(jobs), label)
    return results


@lru_cache(maxsize=1)
def _response_schema() -> types.Schema:
    """Derive a Gemini response schema from the answer template.

    Returns:
        A schema requiring every template key, so the model cannot omit one.
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
    """Build the schema fragment describing one template value.

    Args:
        value: A template leaf, list or nested object.
        path: Dotted path to the value, used in error messages.

    Returns:
        The schema fragment for that value, with a closed set of options
        where the template spells one out.
    """
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
    if isinstance(value, bool):
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
        return types.Schema(type=types.Type.STRING, description=value or None)

    LOG.warning(
        "Template field %r is null, so it carries no type or guidance. "
        "Replace it with a description, a literal like 0, or a '<int> ...' tag.",
        where,
    )
    return types.Schema(type=types.Type.STRING)


def _enum_options(value: str) -> list[str] | None:
    """Read ``"A | B | C"`` as a closed set of allowed values.

    Args:
        value: A template description, which may list alternatives.

    Returns:
        The alternatives if the description is a clean enumeration, else None.
    """
    if "|" not in value:
        return None
    options = [part.strip() for part in value.split("|")]
    return options if len(options) >= 2 and all(options) else None


def _parse_json_response(response: Any) -> Any:
    """Turn the model's reply text into Python data.

    Args:
        response: The raw SDK response object.

    Returns:
        The decoded JSON payload.

    Raises:
        GeminiError: The reply was empty, blocked, or not valid JSON.
    """
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, (dict, list)):
        return parsed

    text = getattr(response, "text", None)
    if not text:
        feedback = getattr(response, "prompt_feedback", None)
        raise GeminiError(f"Model returned no text (feedback: {feedback}).")

    cleaned = text.strip()
    if cleaned.startswith("```"):
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
    """Assert the reply matches the template and has no empty fields.

    Args:
        parsed: The decoded reply, or a fragment of it.
        template: The corresponding fragment of the answer template.
        path: Dotted path to the fragment, used in error messages.

    Raises:
        GeminiError: A key is missing or unexpected, a type is wrong, or a
            field came back empty.
    """
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


_MUST_ESTIMATE = re.compile(r"never\s*0", re.I)


def _require_estimates(parsed: Any, template: Any,
                       *, last_attempt: bool, path: str = "") -> None:
    """Reject zeros in fields the template insists on an estimate for.

    A field opts in by saying "never 0" in its template description, which
    keeps the template the single source of truth. Raising sends the listing
    back for another attempt; on the final try the value is kept and logged
    instead, since a missing estimate is a poorer assessment rather than a
    reason to discard the listing.

    Args:
        parsed: The decoded reply, or a fragment of it.
        template: The corresponding fragment of the answer template.
        last_attempt: Whether this is the final attempt for this listing.
        path: Dotted path to the fragment, used in error messages.

    Raises:
        GeminiError: A field requiring an estimate came back 0 and another
            attempt remains.
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
    """Rebuild the reply in the template's key order.

    Args:
        parsed: The validated reply, or a fragment of it.
        template: The corresponding fragment of the answer template.

    Returns:
        The same data with keys in the template's order.
    """
    if isinstance(template, dict) and isinstance(parsed, dict):
        return {k: _ordered_like(parsed[k], v) for k, v in template.items()}
    if isinstance(template, list) and isinstance(parsed, list) and template:
        return [_ordered_like(item, template[0]) for item in parsed]
    return parsed


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    """Return the Gemini client, built on first use.

    Building lazily means importing this module never requires a key.

    Returns:
        A configured SDK client, cached for the life of the process.

    Raises:
        GeminiConfigError: No API key is set in the environment.
    """
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise GeminiConfigError(
            "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com "
            "and export it before running."
        )
    return genai.Client()


@lru_cache(maxsize=1)
def _load_template() -> dict[str, Any]:
    """Read the answer-format template from disk.

    The template is data rather than a module, so it can be edited without
    touching the code.

    Returns:
        The decoded template.

    Raises:
        GeminiConfigError: The file is missing, unreadable or not valid JSON.
    """
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
    """Read the resume, cached on path and modification time.

    Args:
        path_str: Path to the resume file.
        mtime: Its modification time, which busts the cache when it changes.

    Returns:
        The file's raw bytes, so a long run reads the PDF only once.

    Raises:
        GeminiConfigError: The file is missing or larger than the inline cap.
    """
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
    """Wrap the resume as an inline prompt part.

    Args:
        path: Path to the resume PDF.

    Returns:
        A part carrying the PDF bytes, ready to send with the prompt.

    Raises:
        GeminiConfigError: The file is missing or too large to send inline.
    """
    if not path.is_file():
        raise GeminiConfigError(f"Resume not found at {path}.")
    data = _resume_bytes(str(path), path.stat().st_mtime)
    return types.Part.from_bytes(data=data, mime_type="application/pdf")


def _to_dict(job: JobListing | dict[str, Any]) -> dict[str, Any]:
    """Render a listing as plain fields for the prompt.

    Args:
        job: A JobListing or an equivalent mapping.

    Returns:
        A dict of the listing's fields, so the model sees structured data
        rather than a dataclass repr.
    """
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