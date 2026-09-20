"""Delivery of job assessments to a Telegram chat via the Bot API.

Assessment dictionaries are rendered as readable HTML, split across several
messages when they exceed Telegram's size cap, and posted to a chat with an
optional inline button linking back to the original listing.

Example:
    from telegram_client import send_job_assessment

    send_job_assessment(chat_id, entry)

Environment:
    TELEGRAM_BOT_TOKEN: bot token issued by @BotFather. Required.

To find a chat id, message the bot once and read the update feed at
``https://api.telegram.org/bot<TOKEN>/getUpdates``.

Note:
    The Bot API offers no idempotency key, so a send whose outcome is
    unknown is never retried here. See ``TelegramUncertain``.
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

MAX_MESSAGE_CHARS = 4000

SEND_DELAY = 1.0
MAX_RETRIES = 3


class TelegramError(RuntimeError):
    """Raised when the Telegram API rejects a request."""


class TelegramUncertain(TelegramError):
    """Raised when a send reached Telegram but its outcome is unknown.

    A read timeout or a 5xx arrives after the request was delivered, so
    Telegram may already have posted the message. Because the Bot API has no
    idempotency key, a retry would post it a second time. The send is
    abandoned instead and the caller decides: treating it as sent risks losing
    one message, retrying it risks a duplicate.
    """


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
    """Post a message to a chat, splitting it if it is too long.

    Args:
        chat_id: Target chat, as a numeric id or an ``@channelname``.
        message: A JSON-shaped mapping or sequence, rendered as readable
            text, or a plain string, sent as-is.
        title: Bold heading placed above the body. Optional.
        labels: Field-name overrides, e.g. ``{"match_score": "Match"}``.
        url_button: A ``(text, url)`` pair rendered as a tappable button
            beneath the final chunk.
        as_code: Send the raw JSON in a code block instead of formatted text.
        disable_preview: Suppress link previews. Defaults to True.

    Returns:
        One Telegram result object per message sent. A body short enough to
        fit in a single message yields a one-element list.

    Raises:
        TelegramError: The API rejected the request or the token is unset.
        TelegramUncertain: A chunk may or may not have been posted.
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
    """Post a single assessed listing.

    The heading combines the job title with the hiring company, and an "Open
    listing" button is attached when the entry carries a URL.

    Args:
        chat_id: Target chat, as a numeric id or an ``@channelname``.
        entry: A ``{"job": ..., "assessment": ...}`` mapping as produced by
            ``gemini_client.analyze_jobs``. A bare assessment is also accepted.
        labels: Field-name overrides passed through to ``send_message``.

    Returns:
        One Telegram result object per message sent.

    Raises:
        TelegramError: The API rejected the request or the token is unset.
        TelegramUncertain: The message may or may not have been posted.
    """
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
    """Post several assessed listings in turn.

    Args:
        chat_id: Target chat, as a numeric id or an ``@channelname``.
        entries: Assessed entries from ``gemini_client.analyze_jobs``.
        labels: Field-name overrides passed through to ``send_message``.
        skip_failures: Log and continue past a failed send rather than
            aborting the batch. Defaults to True.

    Returns:
        How many entries were posted successfully.

    Raises:
        TelegramError: A send failed and ``skip_failures`` is False.
    """
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


def _build_text(
    message: Any,
    *,
    title: str | None,
    labels: Mapping[str, str] | None,
    as_code: bool,
) -> str:
    """Render a message body as Telegram-flavoured HTML.

    Args:
        message: The payload to render.
        title: Bold heading placed above the body, or None.
        labels: Field-name overrides, or None.
        as_code: Render the raw JSON in a ``<pre>`` block instead.

    Returns:
        The HTML body, ready to be split and sent.
    """
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
    """Flatten JSON-shaped data into indented HTML lines.

    Args:
        data: Mapping, sequence or scalar to render.
        labels: Field-name overrides keyed by the original field name.
        depth: Current nesting level, used for indentation.

    Returns:
        One rendered line per field or list item.
    """
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
    """Report whether a value should be rendered as a list.

    Args:
        value: The value to test.

    Returns:
        True for lists and tuples, which render as bullets.
    """
    return isinstance(value, (list, tuple))


def _label(key: str) -> str:
    """Turn a field name into a human-readable label.

    Args:
        key: A snake_case field name, e.g. ``salary_min_ils_monthly``.

    Returns:
        The label, e.g. ``Salary min ils monthly``.
    """
    text = str(key).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else str(key)


def _scalar(value: Any) -> str:
    """Render a single value for display.

    Args:
        value: Any scalar drawn from an assessment.

    Returns:
        Its display form: an em dash for None, yes/no for booleans, and
        ``str(value)`` otherwise.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _split_text(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split rendered text into sendable chunks.

    Splitting happens on line boundaries so that HTML tags opened and closed
    within a line are never separated. A single line longer than the limit is
    cut at the limit.

    Args:
        text: The rendered HTML body.
        limit: Maximum characters per chunk. Defaults to MAX_MESSAGE_CHARS.

    Returns:
        Non-empty chunks in order. Text within the limit yields one chunk.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for line in text.split("\n"):
        while len(line) > limit:
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


_session: requests.Session | None = None


def _token() -> str:
    """Read the bot token from the environment.

    Returns:
        The token issued by @BotFather.

    Raises:
        TelegramError: TELEGRAM_BOT_TOKEN is unset or empty.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise TelegramError(
            "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and "
            "export the token it gives you."
        )
    return token


def _get_session() -> requests.Session:
    """Return the shared HTTP session, creating it on first use.

    Returns:
        A module-level session, so connections are reused across sends.
    """
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _call(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to the Bot API, retrying only when nothing can have been sent.

    Retrying a send whose outcome is unknown is what posts a message twice, so
    only two failures are repeated here: one where the connection was never
    established, and 429, where Telegram states outright that it did not
    accept the request.

    Args:
        method: Bot API method name, e.g. ``sendMessage``.
        payload: JSON body for the call.

    Returns:
        The ``result`` object from the API response.

    Raises:
        TelegramError: The token is unset, the API rejected the request, or
            the connection could not be established after MAX_RETRIES tries.
        TelegramUncertain: The request reached Telegram but no usable reply
            came back, so the message may or may not have been posted.
    """
    url = f"{API_ROOT}/bot{_token()}/{method}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _get_session().post(
                url, json=payload, timeout=REQUEST_TIMEOUT
            )
        except requests.ConnectionError as exc:
            if attempt == MAX_RETRIES:
                raise TelegramError(f"{method} failed: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        except requests.Timeout as exc:
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
            raise TelegramError(
                f"{method} rejected ({body.get('error_code')}): "
                f"{body.get('description')}"
            )
        return body.get("result", {})

    raise TelegramError(f"{method} failed after {MAX_RETRIES} attempts.")


def _retry_after(response: requests.Response) -> float:
    """Extract how long to wait after a rate-limit response.

    Args:
        response: A 429 response from the Bot API.

    Returns:
        Seconds to wait, taken from the API's ``retry_after`` parameter,
        falling back to the Retry-After header and then to five seconds.
    """
    try:
        return float(response.json()["parameters"]["retry_after"])
    except (ValueError, KeyError, TypeError):
        return float(response.headers.get("Retry-After", 5))