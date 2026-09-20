#!/usr/bin/env python3
"""Job-hunting agent: search, assess, filter, notify, once every 24 hours.

Takes no arguments; everything is configured in the Parameters block below.
Each cycle searches the configured sites, drops listings already sent,
assesses what is left against the CV, applies the filter rules, and sends
the survivors to Telegram.

Example:
    export GEMINI_API_KEY=... TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
    python agent.py

Environment:
    GEMINI_API_KEY: Gemini API credentials. Required.
    TELEGRAM_BOT_TOKEN: bot token from @BotFather. Required unless DRY_RUN.
    TELEGRAM_CHAT_ID: chat to notify. Required unless DRY_RUN.

Sites:
    AllJobs is the largest Israeli board. Its guest search page takes a
    free-text parameter, ``freetxt``, which is what the site's own search
    box drives, so SEARCH_TERMS reaches it directly. That match runs
    against the whole advert rather than the title alone, so it widens the
    net; the ``type`` filters narrow it, covering all listings, those
    suitable for students, and those needing no experience.

    GotFriends is a hi-tech placement agency carrying startup roles, many
    of them exclusive. Its board takes only a page number, so its entries
    are categories rather than queries and carry no ``{query}``; job_search
    fetches those once per cycle instead of once per term.

    Entries sharing a name pool into one bucket, and a site's quota is
    split evenly across its URLs, so no single URL can spend it all.
    AllJobs type ids and GotFriends category paths come from the sites'
    own URLs: apply the filter there and copy it out of the address bar.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:
    dotenv_values = load_dotenv = None

from gemini_client import TEMPLATE_PATH, analyze_jobs
from job_filtering import JobFilter, JobHistory, filter_jobs, job_id
from job_search import (SiteConfig, alljobs_listings,
                        gotfriends_listings, search_jobs)
from telegram_client import TelegramUncertain, send_job_assessment

LOG = logging.getLogger("agent")

_ALLJOBS = ("https://www.alljobs.co.il/SearchResultsGuest.aspx"
            "?freetxt={query}&page={page}&position=&type=%s&city=&region=")
_GOTFRIENDS = "https://www.gotfriends.co.il/jobslobby/%s/?page={page}"

SITES: list[SiteConfig] = [
    SiteConfig(name="AllJobs", search_url=_ALLJOBS % "",
               parser=alljobs_listings),
    SiteConfig(name="AllJobs", search_url=_ALLJOBS % "14",
               parser=alljobs_listings),
    SiteConfig(name="AllJobs", search_url=_ALLJOBS % "33",
               parser=alljobs_listings),

    SiteConfig(name="GotFriends", search_url=_GOTFRIENDS % "software/cplusplus-programmer",
               parser=gotfriends_listings),
    SiteConfig(name="GotFriends", search_url=_GOTFRIENDS % "software/real-time-engineerembedded-engineer",
               parser=gotfriends_listings),
    SiteConfig(name="GotFriends", search_url=_GOTFRIENDS % "system/linux-system",
               parser=gotfriends_listings),
    SiteConfig(name="GotFriends", search_url=_GOTFRIENDS % "software/graduate-with-high-honors",
               parser=gotfriends_listings),
]

SEARCH_TERMS = [
    "student software developer",
    "junior C++ developer",
    "junior Linux developer",
    "junior embedded developer",
    "entry level systems programmer",
    "student backend developer",
    "junior cloud engineer",
    "junior infrastructure engineer",
]

JOB_FIELD = "A student or entry-level position in Low-Level Systems or Cloud Infrastructure"
RESUME_PATH = Path("resume.pdf")
HOME_LOCATION = "Shefayim, Israel"

LISTINGS_PER_CYCLE = 70
HISTORY_SIZE = 700
INTERVAL_HOURS = 24.0
ENV_FILE = Path(".env")
HISTORY_FILE = Path("job_history.json")
FILTER_CONFIG = Path("job_filtering.json")

DRY_RUN = False
RUN_ONCE = False
LOG_LEVEL = logging.INFO

_stop = threading.Event()


def telegram_chat_id() -> str:
    """Return the chat to notify, from .env or the shell.

    Deliberately a function rather than a constant: ``.env`` is loaded inside
    ``main``, long after this module is imported, so a module-level lookup
    would always read an empty value.

    Returns:
        The chat id, or an empty string when it is unset.
    """
    return os.environ.get("TELEGRAM_CHAT_ID", "").strip()


def run_cycle(job_filter: JobFilter, history: JobHistory) -> int:
    """Run the pipeline once.

    Listings already in the history are dropped before any assessment is
    paid for, and listings reached by more than one term or site are
    collapsed. Survivors are recorded only once Telegram has accepted them,
    so a failed send is retried on the next cycle.

    Args:
        job_filter: The rules deciding which assessments are worth sending.
        history: Ids already sent, read to skip and written on success.

    Returns:
        How many messages were sent.
    """
    LOG.info("Searching %d site(s) for %d term(s)...",
             len(SITES), len(SEARCH_TERMS))
    listings = search_jobs(SITES, SEARCH_TERMS, limit=LISTINGS_PER_CYCLE)
    LOG.info("Found %d listing(s).", len(listings))
    if not listings:
        return 0

    fresh, seen_here = [], set()
    old = duplicates = 0
    for listing in listings:
        identifier = _listing_id(listing)
        if identifier is None:
            fresh.append(listing)
            continue
        if identifier in history:
            old += 1
            continue
        if identifier in seen_here:
            duplicates += 1
            continue
        seen_here.add(identifier)
        fresh.append(listing)

    if old:
        LOG.info("Skipping %d listing(s) already in history.", old)
    if duplicates:
        LOG.info("Collapsed %d duplicate listing(s).", duplicates)
    if not fresh:
        return 0

    LOG.info("Assessing %d listing(s) with Gemini...", len(fresh))
    results = analyze_jobs(
        RESUME_PATH,
        JOB_FIELD,
        fresh,
        home_location=HOME_LOCATION or None,
    )
    LOG.info("Got %d assessment(s).", len(results))
    if not results:
        return 0

    keepers = filter_jobs(
        results, job_filter=job_filter, history=history, remember=False
    )
    LOG.info("%d of %d passed the filter.", len(keepers), len(results))

    sent = 0
    handled: set[str] = set()
    for entry in keepers:
        title = entry["assessment"].get("job_title", "job")
        score = entry["assessment"].get("match_score", "?")
        identifier = _listing_id(entry)

        if identifier and (identifier in history or identifier in handled):
            LOG.debug("Skipping duplicate in this batch: %s", title)
            continue

        if DRY_RUN:
            LOG.info("[dry run] would send: %s (score %s)", title, score)
            LOG.debug("%s", json.dumps(entry["assessment"], indent=2,
                                       ensure_ascii=False))
            if identifier:
                handled.add(identifier)
            continue

        try:
            send_job_assessment(telegram_chat_id(), entry)
        except TelegramUncertain as exc:
            LOG.warning("Unsure whether %s was sent (%s); recording it as "
                        "sent so it cannot be posted twice.", title, exc)
            if identifier:
                history.add(identifier)
                handled.add(identifier)
            continue
        except Exception as exc:
            LOG.warning("Could not send %s: %s", title, exc)
            continue

        if identifier:
            history.add(identifier)
            handled.add(identifier)
        else:
            LOG.warning("Sent %s but could not derive an ID; it may repeat.", title)
        sent += 1
        LOG.info("Sent: %s (score %s)", title, score)

    return sent


def _listing_id(item) -> str | None:
    """Derive an id for a raw listing or an assessed entry.

    Both shapes route through ``job_filtering.job_id``, which hashes the
    listing's company, content and location and never the assessment, so the
    checks before and after the model runs agree on identity.

    Args:
        item: A JobListing, an assessed entry, or a listing mapping.

    Returns:
        The id, or None when the job carries no identifying fields.
    """
    if is_dataclass(item):
        payload = {"job": asdict(item)}
    elif isinstance(item, dict) and ("job" in item or "assessment" in item):
        payload = item
    else:
        payload = {"job": dict(item)}
    try:
        return job_id(payload)
    except ValueError:
        return None


def load_env() -> None:
    """Copy ``.env`` into the environment, if there is one.

    Python does not read ``.env`` by itself and the client modules only look
    at ``os.environ``, so without this step the file is inert. Variables
    already exported in the shell win. Missing files are ignored; a missing
    python-dotenv or a malformed key is reported and the run continues.
    """
    if not ENV_FILE.is_file():
        return

    if load_dotenv is None:
        LOG.error("%s exists but python-dotenv is not installed, so it is "
                  "being ignored. Run: pip install python-dotenv", ENV_FILE)
        return

    quoted = [k for k in dotenv_values(ENV_FILE) if k != k.strip("\"'")]
    if quoted:
        LOG.error("These keys in %s are wrapped in quotes, which makes the "
                  "quotes part of the name: %s. Write them as KEY=value.",
                  ENV_FILE, ", ".join(quoted))

    load_dotenv(ENV_FILE, override=False)
    LOG.info("Loaded %s", ENV_FILE)


def check_configuration() -> list[str]:
    """Check the configuration before the first cycle.

    Catches missing pieces now rather than 24 hours from now: empty site or
    term lists, absent resume, filter config or answer template, nonsensical
    numbers, and unset credentials. Telegram credentials are only required
    when DRY_RUN is off.

    Returns:
        One message per problem found; empty when the setup is complete.
    """
    problems: list[str] = []

    if not SITES:
        problems.append("SITES is empty")
    if not SEARCH_TERMS:
        problems.append("SEARCH_TERMS is empty")
    if not RESUME_PATH.is_file():
        problems.append(f"Resume not found at {RESUME_PATH}")
    if LISTINGS_PER_CYCLE < 1:
        problems.append("LISTINGS_PER_CYCLE must be at least 1")
    if HISTORY_SIZE < 1:
        problems.append("HISTORY_SIZE must be at least 1")
    if INTERVAL_HOURS <= 0:
        problems.append("INTERVAL_HOURS must be positive")
    if not FILTER_CONFIG.is_file():
        problems.append(f"Filter config not found at {FILTER_CONFIG}")
    if not TEMPLATE_PATH.is_file():
        problems.append(
            f"Gemini answer template not found at {TEMPLATE_PATH} "
            "(set JOB_TEMPLATE to point elsewhere)"
        )
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        problems.append("GEMINI_API_KEY is not set in the environment")

    if not DRY_RUN:
        if not telegram_chat_id():
            problems.append("TELEGRAM_CHAT_ID is not set; add it to .env")
        if not os.environ.get("TELEGRAM_BOT_TOKEN"):
            problems.append("TELEGRAM_BOT_TOKEN is not set in the environment")

    return problems


def _handle_signal(signum, _frame) -> None:
    """Ask the run to stop at the end of the current cycle.

    Args:
        signum: The signal received.
        _frame: The interrupted stack frame. Unused.
    """
    LOG.info("Received %s; finishing up.", signal.Signals(signum).name)
    _stop.set()


def main() -> int:
    """Configure logging, validate the setup, and run cycles until stopped.

    Returns:
        A process exit status: 0 on a clean stop, 1 when the configuration
        is incomplete.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    load_env()

    problems = check_configuration()
    if problems:
        for problem in problems:
            LOG.error("%s", problem)
        return 1

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    job_filter = JobFilter(FILTER_CONFIG)
    history = JobHistory(size=HISTORY_SIZE, path=HISTORY_FILE)
    LOG.info("Starting with %d job(s) in history (capacity %d).",
             len(history), history.size)
    if DRY_RUN:
        LOG.info("Dry run: nothing will be sent to Telegram.")

    cycle = 0
    while not _stop.is_set():
        cycle += 1
        LOG.info("--- cycle %d ---", cycle)
        started = time.monotonic()
        try:
            sent = run_cycle(job_filter, history)
            LOG.info("Cycle %d done in %.0fs; %d message(s) sent.",
                     cycle, time.monotonic() - started, sent)
        except Exception:
            LOG.exception("Cycle %d failed.", cycle)

        if RUN_ONCE or _stop.is_set():
            break

        seconds = INTERVAL_HOURS * 3600
        next_run = datetime.now() + timedelta(seconds=seconds)
        LOG.info("Sleeping until %s.", next_run.strftime("%Y-%m-%d %H:%M"))
        if _stop.wait(seconds):
            break

    LOG.info("Stopped after %d cycle(s).", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())