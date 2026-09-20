"""Decide which assessed jobs are worth sending, and remember what was sent.

Two concerns live here. ``JobFilter`` checks an assessment against rules
declared in ``job_filtering.json``, collecting every reason a job fails
rather than stopping at the first. ``JobHistory`` keeps a bounded, on-disk
record of the job ids already delivered, so the same posting is never sent
twice.

Example:
    from job_filtering import JobFilter, JobHistory, filter_jobs

    keepers = filter_jobs(results, history=history)

Identity is derived by ``job_id``, which hashes the scraped listing rather
than the assessment, so the same posting keeps one id whether it is seen
before or after the model has judged it.
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

UNKNOWN_STRINGS = {"", "not specified", "unspecified", "unknown", "n/a", "na", "-", "—"}

COMPANY_KEYS = ("company", "company_name", "hiring_company")
LOCATION_KEYS = ("location", "job_location", "city")
CONTENT_KEYS = ("description", "summary", "snippet", "content", "title", "job_title")


class FilterConfigError(ValueError):
    """Raised when job_filtering.json cannot be understood."""


@dataclass
class FilterResult:
    """The verdict on one job, with every reason it was dropped.

    Attributes:
        job_id: The job's stable id, or None when one cannot be derived.
        passed: Whether the job survived every rule.
        reasons: Human-readable reasons it was dropped; empty when it passed.
    """
    job_id: str | None
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Render the verdict as one log-friendly line.

        Returns:
            The id followed by either "kept" or the reasons it was dropped.
        """
        verdict = "pass" if self.passed else "drop"
        return f"[{verdict}] {self.job_id or '<no id>'}" + (
            " — " + "; ".join(self.reasons) if self.reasons else ""
        )


class JobHistory:
    """The last N job ids that were sent, oldest dropped first.

    Backed by a JSON file and usable like a set with a capacity: membership
    testing, iteration and ``len`` all work directly on it.

    Attributes:
        size: How many ids are remembered before the oldest is evicted.
        path: The JSON file backing the history, or None when in memory only.
        autosave: Whether every change is written to disk immediately.
    """

    def __init__(
        self,
        size: int,
        path: str | Path | None = HISTORY_PATH,
        *,
        autosave: bool = True,
    ) -> None:
        """Open a history file, creating an empty history if it is absent.

        Args:
            size: How many ids to remember. Older ids are dropped first.
            path: JSON file backing the history, or None to keep it in memory.
            autosave: Write to disk on every change. Defaults to True.

        Raises:
            ValueError: ``size`` is less than one.
        """
        if size < 1:
            raise ValueError("History size must be at least 1.")
        self.size = size
        self.path = Path(path) if path is not None else None
        self.autosave = autosave
        self._ids: deque[str] = deque(maxlen=size)
        self._seen: set[str] = set()
        self.load()

    def __contains__(self, job_id: object) -> bool:
        """Report whether an id has already been sent.

        Args:
            job_id: The id to look for.

        Returns:
            True if the id is in the history.
        """
        return job_id in self._seen

    def __len__(self) -> int:
        """Return how many ids are currently remembered."""
        return len(self._ids)

    def __iter__(self):
        """Iterate the remembered ids, oldest first."""
        return iter(self._ids)

    def __repr__(self) -> str:
        """Return a short debugging summary of size, capacity and path."""
        return f"JobHistory({len(self._ids)}/{self.size} ids, path={self.path})"

    @property
    def ids(self) -> list[str]:
        """Return the remembered ids, oldest first.

        Returns:
            A list copy, safe to mutate.
        """
        return list(self._ids)

    def add(self, job_id: str) -> bool:
        """Record an id, evicting the oldest when at capacity.

        Args:
            job_id: The id to record.

        Returns:
            False if the id was already present and nothing changed.
        """
        if not job_id:
            raise ValueError("Refusing to store an empty job id.")
        if job_id in self._seen:
            return False

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
        """Record the id of a job entry.

        Args:
            entry: An assessed entry or a raw listing.

        Returns:
            The id that was recorded, or None when one cannot be derived.
        """
        identifier = job_id(entry)
        self.add(identifier)
        return identifier

    def clear(self) -> None:
        """Forget every id, saving if autosave is on."""
        self._ids.clear()
        self._seen.clear()
        if self.autosave:
            self.save()

    def resize(self, size: int) -> None:
        """Change the capacity, keeping the newest ids.

        Args:
            size: The new capacity.

        Raises:
            ValueError: ``size`` is less than one.
        """
        if size < 1:
            raise ValueError("History size must be at least 1.")
        self.size = size
        self._ids = deque(list(self._ids)[-size:], maxlen=size)
        self._seen = set(self._ids)
        if self.autosave:
            self.save()

    def load(self) -> None:
        """Re-read the history from disk, discarding in-memory state.

        A missing or unreadable file leaves the history empty rather than
        raising, so a corrupted file costs the record but not the run.
        """
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

        clean: list[str] = []
        for item in stored:
            if isinstance(item, str) and item and item not in clean:
                clean.append(item)
        self._ids = deque(clean[-self.size:], maxlen=self.size)
        self._seen = set(self._ids)
        LOG.debug("Loaded %d ids from %s", len(self._ids), self.path)

    def save(self) -> None:
        """Write the history to disk atomically.

        The file is written beside its destination and moved into place, so a
        crash mid-write cannot leave a truncated history behind.
        """
        if self.path is None:
            return
        payload = {"size": self.size, "ids": list(self._ids)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            LOG.warning("Could not save history to %s: %s", self.path, exc)
            tmp.unlink(missing_ok=True)


def job_id(entry: Mapping[str, Any]) -> str:
    """Derive a stable id for a job from its company, content and location.

    Args:
        entry: An assessed ``{"job": ..., "assessment": ...}`` entry, a raw
            listing dict, or a bare assessment. The listing is always
            preferred as the source, so the checks before and after the model
            runs agree on identity.

    Returns:
        A ``sha256:`` prefixed digest, stable across cosmetic edits.

    Raises:
        ValueError: Company, content and location are all empty, so hashing
            would give every such job the same identity.
    """
    company, content, location = _identity_fields(entry)
    parts = [_normalize_text(company), _normalize_text(content),
             _normalize_text(location)]

    if not any(parts):
        raise ValueError(
            "Cannot derive a job id: company, content and location are all empty."
        )

    payload = "\x1f".join(parts).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:32]


def _safe_job_id(entry: Mapping[str, Any]) -> str | None:
    """Derive a job id, tolerating jobs that have none.

    Args:
        entry: An assessed entry or a raw listing.

    Returns:
        The id, or None when the job carries no identifying fields.
    """
    try:
        return job_id(entry)
    except ValueError:
        return None


def _identity_fields(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    """Pull the three fields that define a job's identity.

    Args:
        entry: An assessed entry or a raw listing.

    Returns:
        The company, content and location strings, any of which may be empty.
    """
    source = _identity_source(entry)
    return (
        _first_present(source, COMPANY_KEYS),
        _first_present(source, CONTENT_KEYS),
        _first_present(source, LOCATION_KEYS),
    )


def _identity_source(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Choose which half of an entry identifies the job.

    Args:
        entry: An assessed entry or a raw listing.

    Returns:
        The scraped listing when it carries identifying fields, falling back
        to the assessment only when the listing is empty.
    """
    if isinstance(entry, Mapping) and ("job" in entry or "assessment" in entry):
        job = entry.get("job") or {}
        if any(_first_present(job, keys)
               for keys in (COMPANY_KEYS, CONTENT_KEYS, LOCATION_KEYS)):
            return job
        return entry.get("assessment") or {}
    return entry


def _first_present(source: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty value among several keys.

    Args:
        source: The mapping to read.
        keys: Field names to try, in order of preference.

    Returns:
        The first non-empty value found, or an empty string.
    """
    for key in keys:
        value = source.get(key) if isinstance(source, Mapping) else None
        if value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return ""


def _normalize_text(value: Any) -> str:
    """Fold away formatting so cosmetic edits do not change identity.

    Args:
        value: Any field value.

    Returns:
        A lowercased, whitespace-collapsed, punctuation-stripped form.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(cleaned.split())


def _split_entry(entry: Mapping[str, Any]) -> tuple[dict, dict]:
    """Separate an entry into its listing and assessment halves.

    Args:
        entry: An assessed entry, a raw listing, or a bare assessment.

    Returns:
        The listing mapping and the assessment mapping, either of which may
        be empty.
    """
    if "assessment" in entry or "job" in entry:
        return dict(entry.get("job") or {}), dict(entry.get("assessment") or {})
    return {}, dict(entry)


def load_config(config: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Load filter rules from a mapping or a JSON file.

    Args:
        config: A mapping of rules, a path to a JSON file, or None to read
            the default ``job_filtering.json``.

    Returns:
        The decoded configuration.

    Raises:
        FilterConfigError: The file is missing, not valid JSON, or declares a
            rule in a shape the checker cannot apply.
    """
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
    """Checks an assessment against the configured rules.

    Attributes:
        config: The loaded configuration.
        rules: The rule table, keyed by assessment field name.
    """

    def __init__(self, config: Mapping[str, Any] | str | Path | None = None) -> None:
        """Load the rules this filter will apply.

        Args:
            config: A mapping of rules, a path to a JSON file, or None to read
                the default ``job_filtering.json``.

        Raises:
            FilterConfigError: The configuration cannot be understood.
        """
        self.config = load_config(config)
        self.rules: dict[str, Any] = dict(self.config.get("rules", {}))

    def check(
        self,
        entry: Mapping[str, Any],
        history: JobHistory | None = None,
    ) -> FilterResult:
        """Evaluate one job against every rule.

        Every rule is applied, so the result names all the reasons a job was
        dropped rather than only the first.

        Args:
            entry: An assessed entry or a bare assessment.
            history: Already-sent ids to check against. Optional.

        Returns:
            A FilterResult carrying the verdict, the reasons and the job id.
        """
        identifier = _safe_job_id(entry)
        _, assessment = _split_entry(entry)

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
    """Return the entries worth sending.

    Every decision is logged at INFO, so enabling logging shows exactly why
    each job was kept or dropped.

    Args:
        entries: Assessed entries from ``gemini_client.analyze_jobs``.
        job_filter: The filter to apply. A default one is built if omitted.
        history: Already-sent ids, used both to skip and to record.
        remember: Record survivors in ``history`` immediately. Pass False to
            record only once delivery has actually succeeded.

    Returns:
        The entries that passed every rule, in input order.
    """
    job_filter = job_filter or JobFilter()
    kept: list[Mapping[str, Any]] = []

    for entry in entries:
        result = job_filter.check(entry, history)
        LOG.info("%s", result)
        if result.passed:
            kept.append(entry)
            if history is not None and remember and result.job_id:
                history.add(result.job_id)

    return kept


def _check_rule(
    name: str,
    value: Any,
    rule: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> list[str]:
    """Apply one rule to one field.

    Args:
        name: The assessment field being checked.
        value: Its value.
        rule: The rule declaration for that field.
        assessment: The whole assessment, for rules whose bound depends on
            another field.

    Returns:
        A reason per violation; empty when the field passes.
    """
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
    """Resolve a rule's bound, which may depend on another field.

    A bound may be fixed, or declared conditionally::

        "min_by": {"field": "days_per_week",
                   "map": {"2": 10500, "3": 16000},
                   "default": 10500}

    Args:
        rule: The rule declaration.
        key: Which bound to resolve, ``min`` or ``max``.
        assessment: The whole assessment, read for the field a conditional
            bound depends on.

    Returns:
        The bound, or None when the rule declares none. A conditional bound
        whose key is absent from the map falls back to its default.
    """
    if key in rule:
        return _as_number(rule[key])

    conditional = rule.get(f"{key}_by")
    if not isinstance(conditional, Mapping):
        return None

    other = assessment.get(conditional.get("field"))
    mapping = conditional.get("map") or {}
    if other is not None:
        for candidate in _key_variants(other):
            if candidate in mapping:
                return _as_number(mapping[candidate])
    return _as_number(conditional.get("default"))


def _key_variants(value: Any) -> list[str]:
    """List the forms a value may take as a map key.

    Args:
        value: The value to look up.

    Returns:
        Its string form and, for whole numbers, its integer string, so that
        3 and "3" both match.
    """
    variants = [str(value)]
    number = _as_number(value)
    if number is not None and float(number).is_integer():
        variants.append(str(int(number)))
    return variants


def _is_unknown(value: Any, rule: Mapping[str, Any]) -> bool:
    """Report whether a value means "not stated".

    Args:
        value: The field value.
        rule: Its rule, which may set ``zero_is_unknown``.

    Returns:
        True for None, for known placeholder strings, and for 0 when the
        template uses 0 to mean "not stated".
    """
    if value is None:
        return True
    if isinstance(value, str) and _clean(value) in UNKNOWN_STRINGS:
        return True
    if rule.get("zero_is_unknown") and isinstance(value, (int, float)) \
            and not isinstance(value, bool) and value == 0:
        return True
    return False


def _clean(value: Any) -> str:
    """Lowercase and collapse whitespace for comparing rule values.

    Deliberately gentler than ``_normalize_text``: punctuation is kept, so an
    enum value like ``On-site`` still has to match ``On-site``.

    Args:
        value: The value to clean.

    Returns:
        The cleaned string.
    """
    return " ".join(str(value).lower().split()) if value else ""


def _as_number(value: Any) -> float | None:
    """Coerce a value to a number if it plausibly is one.

    Args:
        value: The value to coerce.

    Returns:
        The number, or None when the value is not numeric.
    """
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
    """Format a bound for a reason message.

    Args:
        number: The bound to render.

    Returns:
        An integer-looking string for whole numbers, else the plain value.
    """
    return str(int(number)) if float(number).is_integer() else f"{number:g}"


def _searchable(value: Any) -> str:
    """Flatten a value into lowercase text for substring rules.

    Args:
        value: A scalar or a sequence drawn from an assessment.

    Returns:
        One lowercase string covering every part of the value.
    """
    if isinstance(value, (list, tuple)):
        return " | ".join(_clean(item) for item in value)
    return _clean(value)