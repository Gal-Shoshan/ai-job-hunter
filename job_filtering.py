"""Decide which assessed jobs are worth sending, and remember what was sent.

    from job_filtering import JobFilter, JobHistory, filter_jobs

    history = JobHistory(size=200)
    keepers = filter_jobs(results, history=history)   # results from analyze_jobs
    send_job_assessments(chat_id, keepers)

Two gates: the rules in job_filtering.json, and a rolling history of job IDs
so the same listing is never sent twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import unicodedata

LOG = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("JOB_FILTER_CONFIG", "job_filtering.json"))
HISTORY_PATH = Path(os.environ.get("JOB_HISTORY", "job_history.json"))

# Values that mean "the listing did not say".
UNKNOWN_STRINGS = {"", "not specified", "unspecified", "unknown", "n/a", "na", "-", "—"}

# A job's identity is hashed from these three fields. The first key present
# and non-empty wins, so both a raw JobListing (company/description) and an
# assessment (company_name/description) resolve to the same value.
COMPANY_KEYS = ("company", "company_name", "hiring_company")
LOCATION_KEYS = ("location", "job_location", "city")
# "title" is a last resort: listings scraped without structured data carry no
# description, and without it they would all hash identically. Drop it from
# this tuple to hash on the description alone.
CONTENT_KEYS = ("description", "summary", "snippet", "content", "title", "job_title")


class FilterConfigError(ValueError):
    """Raised when job_filtering.json cannot be understood."""


@dataclass
class FilterResult:
    """Why a job was kept or dropped. job_id is None if it can't be derived."""
    job_id: str | None
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        verdict = "pass" if self.passed else "drop"
        return f"[{verdict}] {self.job_id or '<no id>'}" + (
            " — " + "; ".join(self.reasons) if self.reasons else ""
        )


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #

class JobHistory:
    """The last N job IDs that were sent, oldest dropped first.

        history = JobHistory(size=200)
        if job_id not in history:
            history.add(job_id)

    `size` is required and comes from whoever builds the object, not from
    job_filtering.json. Backed by a JSON file so the bot doesn't re-send
    everything after a restart; pass path=None to keep it in memory only.
    """

    def __init__(
        self,
        size: int,
        path: str | Path | None = HISTORY_PATH,
        *,
        autosave: bool = True,
    ) -> None:
        if size < 1:
            raise ValueError("History size must be at least 1.")
        self.size = size
        self.path = Path(path) if path is not None else None
        self.autosave = autosave
        self._ids: deque[str] = deque(maxlen=size)
        self._seen: set[str] = set()
        self.load()

    # -- queries ----------------------------------------------------------- #

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._seen

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)

    def __repr__(self) -> str:
        return f"JobHistory({len(self._ids)}/{self.size} ids, path={self.path})"

    @property
    def ids(self) -> list[str]:
        """Oldest first."""
        return list(self._ids)

    # -- mutation ---------------------------------------------------------- #

    def add(self, job_id: str) -> bool:
        """Record an ID. Returns False if it was already there."""
        if not job_id:
            raise ValueError("Refusing to store an empty job id.")
        if job_id in self._seen:
            return False

        # deque(maxlen) silently drops the oldest; mirror that in the set.
        evicted = self._ids[0] if len(self._ids) == self.size else None
        self._ids.append(job_id)
        self._seen.add(job_id)
        if evicted is not None:
            self._seen.discard(evicted)
            LOG.debug("History full; forgot %s", evicted)

        if self.autosave:
            self.save()
        return True

    def remember(self, entry: Mapping[str, Any]) -> str:
        """Add the ID of a job entry and return it."""
        identifier = job_id(entry)
        self.add(identifier)
        return identifier

    def clear(self) -> None:
        self._ids.clear()
        self._seen.clear()
        if self.autosave:
            self.save()

    def resize(self, size: int) -> None:
        """Change capacity, keeping the newest IDs."""
        if size < 1:
            raise ValueError("History size must be at least 1.")
        self.size = size
        self._ids = deque(list(self._ids)[-size:], maxlen=size)
        self._seen = set(self._ids)
        if self.autosave:
            self.save()

    # -- persistence ------------------------------------------------------- #

    def load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("Could not read %s (%s); starting empty.", self.path, exc)
            return

        stored = raw.get("ids", []) if isinstance(raw, Mapping) else raw
        if not isinstance(stored, list):
            LOG.warning("%s has an unexpected shape; starting empty.", self.path)
            return

        # Keep the newest `size` entries, dropping duplicates.
        clean: list[str] = []
        for item in stored:
            if isinstance(item, str) and item and item not in clean:
                clean.append(item)
        self._ids = deque(clean[-self.size:], maxlen=self.size)
        self._seen = set(self._ids)
        LOG.debug("Loaded %d ids from %s", len(self._ids), self.path)

    def save(self) -> None:
        """Write atomically, so a crash mid-write can't corrupt the file."""
        if self.path is None:
            return
        # "size" is written for readability; the capacity in force is always
        # the one the caller passed to __init__.
        payload = {"size": self.size, "ids": list(self._ids)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            LOG.warning("Could not save history to %s: %s", self.path, exc)
            tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Job identity
# --------------------------------------------------------------------------- #

def job_id(entry: Mapping[str, Any]) -> str:
    """A stable ID for a job, hashed from company, content and location.

    Accepts an analyze_jobs entry ({"job": ..., "assessment": ...}), a raw
    listing dict, or a bare assessment. The listing is always preferred as
    the source, so the pre-Gemini and post-Gemini checks agree.

    Raises ValueError when all three fields are empty, since hashing nothing
    would give every such job the same identity.
    """
    company, content, location = _identity_fields(entry)
    parts = [_normalize_text(company), _normalize_text(content),
             _normalize_text(location)]

    if not any(parts):
        raise ValueError(
            "Cannot derive a job id: company, content and location are all empty."
        )

    # \x1f cannot survive normalization, so field boundaries are unambiguous
    # and "ab|c" cannot collide with "a|bc".
    payload = "\x1f".join(parts).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:32]


def _safe_job_id(entry: Mapping[str, Any]) -> str | None:
    """job_id, or None when the job carries no identifying fields."""
    try:
        return job_id(entry)
    except ValueError:
        return None


def _identity_fields(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    source = _identity_source(entry)
    return (
        _first_present(source, COMPANY_KEYS),
        _first_present(source, CONTENT_KEYS),
        _first_present(source, LOCATION_KEYS),
    )


def _identity_source(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Prefer the scraped listing; fall back to the assessment only if empty.

    This matters: Gemini rewrites the description into a short summary, so
    hashing the assessment for one copy of a job and the listing for another
    would produce two different IDs for the same posting.
    """
    if isinstance(entry, Mapping) and ("job" in entry or "assessment" in entry):
        job = entry.get("job") or {}
        if any(_first_present(job, keys)
               for keys in (COMPANY_KEYS, CONTENT_KEYS, LOCATION_KEYS)):
            return job
        return entry.get("assessment") or {}
    return entry


def _first_present(source: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = source.get(key) if isinstance(source, Mapping) else None
        if value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return ""


def _normalize_text(value: Any) -> str:
    """Fold away formatting so cosmetic edits don't change a job's identity.

    Unicode-normalizes, casefolds, turns anything that isn't a letter or a
    digit into a space, and collapses runs of whitespace. So "Acme, Inc."
    and "ACME  Inc" both become "acme inc".
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(cleaned.split())


def _split_entry(entry: Mapping[str, Any]) -> tuple[dict, dict]:
    if "assessment" in entry or "job" in entry:
        return dict(entry.get("job") or {}), dict(entry.get("assessment") or {})
    return {}, dict(entry)


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def load_config(config: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Load filter rules from a dict or a JSON file."""
    if isinstance(config, Mapping):
        data = dict(config)
    else:
        path = Path(config) if config is not None else CONFIG_PATH
        if not path.is_file():
            raise FilterConfigError(f"Filter config not found at {path}.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FilterConfigError(f"{path} is not valid JSON: {exc}") from exc

    if "history_size" in data:
        LOG.warning(
            "'history_size' in the filter config is ignored; pass the size to "
            "JobHistory(size=...) instead."
        )
        data = {k: v for k, v in data.items() if k != "history_size"}

    rules = data.get("rules")
    if rules is None:
        # Tolerate a bare {field: rule} mapping.
        rules = {k: v for k, v in data.items() if not k.startswith("_")}
        data = {"rules": rules}

    for field_name, rule in rules.items():
        if not isinstance(rule, Mapping):
            raise FilterConfigError(
                f"Rule for {field_name!r} must be an object such as "
                f'{{"min": 75}} or {{"allowed": ["Suitable"]}}, not {rule!r}. '
                "Prose rules cannot be parsed."
            )
    return data


class JobFilter:
    """Checks an assessment against the configured rules."""

    def __init__(self, config: Mapping[str, Any] | str | Path | None = None) -> None:
        self.config = load_config(config)
        self.rules: dict[str, Any] = dict(self.config.get("rules", {}))

    def check(
        self,
        entry: Mapping[str, Any],
        history: JobHistory | None = None,
    ) -> FilterResult:
        """Evaluate one job. Collects every reason, not just the first."""
        identifier = _safe_job_id(entry)
        _, assessment = _split_entry(entry)

        # No identity: it can still be judged and sent, just not deduplicated.
        if identifier is None:
            LOG.warning("No id for this job; it cannot be deduplicated.")
        elif history is not None and identifier in history:
            return FilterResult(identifier, False, ["already sent"])

        reasons: list[str] = []
        for field_name, rule in self.rules.items():
            if field_name not in assessment:
                if rule.get("required"):
                    reasons.append(f"{field_name} is missing from the assessment")
                continue
            reasons.extend(
                _check_rule(field_name, assessment[field_name], rule, assessment)
            )

        return FilterResult(identifier, not reasons, reasons)


def filter_jobs(
    entries: Iterable[Mapping[str, Any]],
    *,
    job_filter: JobFilter | None = None,
    history: JobHistory | None = None,
    remember: bool = True,
) -> list[Mapping[str, Any]]:
    """Return the entries worth sending, recording them in `history`.

    Every decision is logged at INFO, so turning on logging tells you exactly
    why a job was dropped.
    """
    job_filter = job_filter or JobFilter()
    kept: list[Mapping[str, Any]] = []

    for entry in entries:
        result = job_filter.check(entry, history)
        LOG.info("%s", result)
        if result.passed:
            # A job without an ID is still worth sending; it just can't be
            # recorded, so it may reappear next cycle.
            kept.append(entry)
            if history is not None and remember and result.job_id:
                history.add(result.job_id)

    return kept


# --------------------------------------------------------------------------- #
# Rule evaluation
# --------------------------------------------------------------------------- #

def _check_rule(
    name: str,
    value: Any,
    rule: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []

    if _is_unknown(value, rule):
        if not rule.get("unknown_passes", True):
            reasons.append(f"{name} is not specified")
        return reasons

    minimum = _resolve_bound(rule, "min", assessment)
    maximum = _resolve_bound(rule, "max", assessment)

    if minimum is not None or maximum is not None:
        number = _as_number(value)
        if number is None:
            reasons.append(f"{name} is not numeric ({value!r})")
        else:
            if minimum is not None and number < minimum:
                reasons.append(f"{name} {_fmt(number)} is below the minimum {_fmt(minimum)}")
            if maximum is not None and number > maximum:
                reasons.append(f"{name} {_fmt(number)} is above the maximum {_fmt(maximum)}")

    allowed = rule.get("allowed")
    if allowed and _clean(value) not in {_clean(a) for a in allowed}:
        reasons.append(f"{name} {value!r} is not one of {list(allowed)}")

    blocked = rule.get("blocked")
    if blocked and _clean(value) in {_clean(b) for b in blocked}:
        reasons.append(f"{name} {value!r} is blocked")

    haystack = _searchable(value)
    excludes = rule.get("excludes_any")
    if excludes:
        hits = [t for t in excludes if _clean(t) in haystack]
        if hits:
            reasons.append(f"{name} contains {hits}")

    includes = rule.get("includes_any")
    if includes and not any(_clean(t) in haystack for t in includes):
        reasons.append(f"{name} contains none of {list(includes)}")

    if isinstance(value, (list, tuple)):
        max_items = rule.get("max_items")
        min_items = rule.get("min_items")
        if max_items is not None and len(value) > max_items:
            reasons.append(f"{name} has {len(value)} items, more than {max_items}")
        if min_items is not None and len(value) < min_items:
            reasons.append(f"{name} has {len(value)} items, fewer than {min_items}")

    return reasons


def _resolve_bound(
    rule: Mapping[str, Any],
    key: str,
    assessment: Mapping[str, Any],
) -> float | None:
    """Return a fixed bound, or one that depends on another field.

        "min_by": {"field": "days_per_week",
                   "map": {"2": 10500, "3": 16000},
                   "default": 10500}
    """
    if key in rule:
        return _as_number(rule[key])

    conditional = rule.get(f"{key}_by")
    if not isinstance(conditional, Mapping):
        return None

    other = assessment.get(conditional.get("field"))
    mapping = conditional.get("map") or {}
    if other is not None:
        # Look up by the raw value and by its integer form, so 3 and "3" match.
        for candidate in _key_variants(other):
            if candidate in mapping:
                return _as_number(mapping[candidate])
    return _as_number(conditional.get("default"))


def _key_variants(value: Any) -> list[str]:
    variants = [str(value)]
    number = _as_number(value)
    if number is not None and float(number).is_integer():
        variants.append(str(int(number)))
    return variants


def _is_unknown(value: Any, rule: Mapping[str, Any]) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and _clean(value) in UNKNOWN_STRINGS:
        return True
    # 0 is a real number unless the template uses it as "not stated".
    if rule.get("zero_is_unknown") and isinstance(value, (int, float)) \
            and not isinstance(value, bool) and value == 0:
        return True
    return False


def _clean(value: Any) -> str:
    """Lowercase and collapse whitespace, for comparing rule values.

    Deliberately gentler than _normalize_text: it keeps punctuation, so an
    enum value like "On-site" still has to match "On-site".
    """
    return " ".join(str(value).lower().split()) if value else ""


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _fmt(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else f"{number:g}"


def _searchable(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " | ".join(_clean(item) for item in value)
    return _clean(value)