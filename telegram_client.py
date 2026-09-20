"""Send job assessments to Telegram.

    from telegram_client import send_message, send_job_assessment

    send_message(chat_id, assessment_dict)
    send_job_assessment(chat_id, entry)   # entry from analyze_jobs()

Requires TELEGRAM_BOT_TOKEN in the environment (create a bot via @BotFather).
To find your chat_id, message your bot once and read:
    https://api.telegram.org/bot<TOKEN>/getUpdates
"""

from __future__ import annotations

import json
import logging
import os
import time
from html import escape
from typing import Any, Iterable, Mapping, Sequence

import requests

LOG = logging.getLogger(__name__)

API_ROOT = "https://api.telegram.org"
REQUEST_TIMEOUT = 20

# Telegram caps a text message at 4096 UTF-16 code units; leave room for
# the header a split adds.
MAX_MESSAGE_CHARS = 4000

# Telegram tolerates roughly one message per second to a given chat.
SEND_DELAY = 1.0
MAX_RETRIES = 3


class TelegramError(RuntimeError):
    """Raised when the Telegram API rejects a request."""


class TelegramUncertain(TelegramError):
    """Raised when a send reached Telegram but the outcome is unknown.

    A read timeout or a 5xx arrives *after* the request was delivered, so
    Telegram may well have posted the message already. The Bot API has no
    idempotency key, which means a retry would post it a second time. The
    send is abandoned instead and the caller decides: treating it as sent
    risks losing one message, retrying it risks a duplicate.
    """


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def send_message(
    chat_id: int | str,
    message: Mapping[str, Any] | Sequence[Any] | str,
    *,
    title: str | None = None,
    labels: Mapping[str, str] | None = None,
    url_button: tuple[str, str] | None = None,
    as_code: bool = False,
    disable_preview: bool = True,
) -> list[dict[str, Any]]:
    """Send `message` to `chat_id` and return the Telegram result objects.

    message     a JSON-shaped dict or list (rendered as readable text), or a
                plain string (sent as-is).
    title       optional bold heading placed above the body.
    labels      override field names, e.g. {"match_score": "Match"}.
    url_button  (text, url) rendered as a tappable button under the message.
    as_code     send the raw JSON in a code block instead of formatted text.

    Long messages are split across several sends; one result dict is
    returned per send.
    """
    text = _build_text(message, title=title, labels=labels, as_code=as_code)
    chunks = _split_text(text)

    results: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": bool(disable_preview)},
        }
        # Attach the button only to the final chunk, where it belongs.
        if url_button and index == len(chunks) - 1:
            label, url = url_button
            payload["reply_markup"] = {
                "inline_keyboard": [[{"text": label, "url": url}]]
            }
        results.append(_call("sendMessage", payload))
        if index < len(chunks) - 1:
            time.sleep(SEND_DELAY)
    return results


def send_job_assessment(
    chat_id: int | str,
    entry: Mapping[str, Any],
    *,
    labels: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Send one {"job": ..., "assessment": ...} entry from analyze_jobs()."""
    job = dict(entry.get("job") or {})
    assessment = dict(entry.get("assessment") or entry)

    heading = (
        assessment.get("job_title")
        or job.get("title")
        or "Job match"
    )
    company = assessment.get("company_name") or job.get("company")
    if company and company.lower() != "not specified":
        heading = f"{heading} — {company}"

    url = job.get("url") or assessment.get("url")
    return send_message(
        chat_id,
        assessment,
        title=heading,
        labels=labels,
        url_button=("Open listing", url) if url else None,
    )


def send_job_assessments(
    chat_id: int | str,
    entries: Iterable[Mapping[str, Any]],
    *,
    labels: Mapping[str, str] | None = None,
    skip_failures: bool = True,
) -> int:
    """Send a batch of assessments. Returns how many went out."""
    sent = 0
    for entry in entries:
        try:
            send_job_assessment(chat_id, entry, labels=labels)
            sent += 1
        except Exception as exc:
            LOG.warning("Could not send assessment: %s", exc)
            if not skip_failures:
                raise
        time.sleep(SEND_DELAY)
    return sent


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _build_text(
    message: Any,
    *,
    title: str | None,
    labels: Mapping[str, str] | None,
    as_code: bool,
) -> str:
    if isinstance(message, str):
        body = escape(message)
    elif as_code:
        dumped = json.dumps(message, indent=2, ensure_ascii=False)
        body = f"<pre>{escape(dumped)}</pre>"
    else:
        body = "\n".join(_render(message, labels or {}))

    if title:
        return f"<b>{escape(str(title))}</b>\n\n{body}"
    return body


def _render(data: Any, labels: Mapping[str, str], depth: int = 0) -> list[str]:
    """Turn JSON data into readable HTML lines."""
    pad = "  " * depth
    lines: list[str] = []

    if isinstance(data, Mapping):
        for key, value in data.items():
            label = escape(labels.get(key) or _label(key))
            if isinstance(value, Mapping) or _is_seq(value):
                lines.append(f"{pad}<b>{label}</b>")
                lines.extend(_render(value, labels, depth + 1))
            else:
                lines.append(f"{pad}<b>{label}:</b> {escape(_scalar(value))}")
        return lines

    if _is_seq(data):
        items = list(data)
        if not items:
            return [f"{pad}—"]
        for item in items:
            if isinstance(item, Mapping) or _is_seq(item):
                lines.extend(_render(item, labels, depth + 1))
            else:
                lines.append(f"{pad}• {escape(_scalar(item))}")
        return lines

    return [f"{pad}{escape(_scalar(data))}"]


def _is_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _label(key: str) -> str:
    """salary_min_ils_monthly -> Salary min ils monthly"""
    text = str(key).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else str(key)


def _scalar(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _split_text(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split on line boundaries, keeping HTML tags intact within a line."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for line in text.split("\n"):
        while len(line) > limit:  # a single oversized line
            if current:
                chunks.append("\n".join(current))
                current, length = [], 0
            chunks.append(line[:limit])
            line = line[limit:]
        if length + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1

    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

_session: requests.Session | None = None


def _token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise TelegramError(
            "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and "
            "export the token it gives you."
        )
    return token


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _call(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to the Bot API, retrying only when nothing can have been sent.

    Retrying a send whose outcome is unknown is what posts a message twice,
    so only two failures are retried here: one where the connection was
    never established, and 429, where Telegram states outright that it did
    not accept the request. A read timeout or a 5xx is reported as
    TelegramUncertain instead of being repeated.
    """
    url = f"{API_ROOT}/bot{_token()}/{method}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _get_session().post(
                url, json=payload, timeout=REQUEST_TIMEOUT
            )
        except requests.ConnectionError as exc:
            # No connection, so the request never arrived: safe to repeat.
            if attempt == MAX_RETRIES:
                raise TelegramError(f"{method} failed: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        except requests.Timeout as exc:
            # Sent, but the reply never came. Telegram may have posted it.
            raise TelegramUncertain(
                f"{method} timed out after the request was sent; it may "
                f"have gone through: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise TelegramError(f"{method} failed: {exc}") from exc

        if response.status_code == 429:
            wait = _retry_after(response)
            LOG.info("Rate limited; waiting %.1fs", wait)
            time.sleep(wait)
            continue

        if response.status_code >= 500:
            raise TelegramUncertain(
                f"{method} returned {response.status_code}; Telegram may "
                "have posted the message before failing."
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise TelegramError(
                f"{method} returned non-JSON ({response.status_code})."
            ) from exc

        if not body.get("ok"):
            # The token never appears here, only in the URL.
            raise TelegramError(
                f"{method} rejected ({body.get('error_code')}): "
                f"{body.get('description')}"
            )
        return body.get("result", {})

    raise TelegramError(f"{method} failed after {MAX_RETRIES} attempts.")


def _retry_after(response: requests.Response) -> float:
    try:
        return float(response.json()["parameters"]["retry_after"])
    except (ValueError, KeyError, TypeError):
        return float(response.headers.get("Retry-After", 5))