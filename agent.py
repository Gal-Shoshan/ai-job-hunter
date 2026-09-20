#!/usr/bin/env python3
"""Job-hunting agent: search, assess, filter, notify — once every 24 hours.

    export GEMINI_API_KEY=...  TELEGRAM_BOT_TOKEN=...
    python agent.py

Takes no arguments. Everything is configured in the Parameters block below.

Each cycle:
  1. search the configured sites          (job_search)
  2. drop listings already sent           (job_filtering history)
  3. assess what's left against the CV    (gemini_client)
  4. apply the filter rules               (job_filtering)
  5. send the survivors to Telegram       (telegram_client)
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
except ImportError:                      # optional; only needed for .env
    dotenv_values = load_dotenv = None

from gemini_client import TEMPLATE_PATH, analyze_jobs
from job_filtering import JobFilter, JobHistory, filter_jobs, job_id
from job_search import (SiteConfig, alljobs_listings,
                        gotfriends_listings, search_jobs)
from telegram_client import TelegramUncertain, send_job_assessment

LOG = logging.getLogger("agent")

# --------------------------------------------------------------------------- #
# Parameters — edit these
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Sites
#
#   AllJobs     the largest Israeli board. SearchResultsGuest.aspx takes a
#               free-text parameter, freetxt, which is what the site's own
#               "חיפוש משרות חופשי" box drives, so SEARCH_TERMS reaches it
#               directly. It matches anywhere in the ad rather than in the
#               title alone, so it widens the net rather than narrowing it;
#               the type= filters below do the narrowing. Each term is run
#               against all listings, against student-suitable ones, and
#               against no-experience ones.
#   GotFriends  a hi-tech placement agency: startup roles, many exclusive.
#               /jobs/ takes only ?page=, so its entries are categories
#               rather than queries and carry no {query} — job_search
#               fetches those once per cycle instead of once per term.
# Entries sharing a name pool into one bucket, and a site's quota is split
# evenly across its URLs, so no single URL can spend it all.
#
# AllJobs type ids and GotFriends category paths come from the sites' own
# URLs: apply the filter there and copy it out of the address bar.
# --------------------------------------------------------------------------- #

_ALLJOBS = ("https://www.alljobs.co.il/SearchResultsGuest.aspx"
            "?freetxt={query}&page={page}&position=&type=%s&city=&region=")
_GOTFRIENDS = "https://www.gotfriends.co.il/jobslobby/%s/?page={page}"

SITES: list[SiteConfig] = [
    SiteConfig(name="AllJobs", search_url=_ALLJOBS % "",
               parser=alljobs_listings),          # every listing
    SiteConfig(name="AllJobs", search_url=_ALLJOBS % "14",
               parser=alljobs_listings),          # מתאים גם לסטודנטים
    SiteConfig(name="AllJobs", search_url=_ALLJOBS % "33",
               parser=alljobs_listings),          # ללא ניסיון

    SiteConfig(name="GotFriends", search_url=_GOTFRIENDS % "software/cplusplus-programmer",
               parser=gotfriends_listings),
    SiteConfig(name="GotFriends", search_url=_GOTFRIENDS % "software/real-time-engineerembedded-engineer",
               parser=gotfriends_listings),
    SiteConfig(name="GotFriends", search_url=_GOTFRIENDS % "system/linux-system",
               parser=gotfriends_listings),
    SiteConfig(name="GotFriends", search_url=_GOTFRIENDS % "software/graduate-with-high-honors",
               parser=gotfriends_listings),
]

# Plain, natural-language phrases. Each is sent to every site exactly as
# written, so keep them short: boards match the words they are given, and a
# long phrase quietly matches nothing. One idea per term; add more terms
# rather than more words.
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

LISTINGS_PER_CYCLE = 70       # listings to pull PER SITE before assessing;
                              # 2 sites x 70 = up to 140 Gemini assessments
HISTORY_SIZE = 700            # how many sent jobs to remember
INTERVAL_HOURS = 24.0         # pause between cycles
ENV_FILE = Path(".env")       # optional; secrets may also come from the shell
HISTORY_FILE = Path("job_history.json")
FILTER_CONFIG = Path("job_filtering.json")

DRY_RUN = False               # True: do everything except send to Telegram
RUN_ONCE = False              # True: one cycle, then exit
LOG_LEVEL = logging.INFO      # logging.DEBUG for per-job detail

# Set by SIGINT/SIGTERM so a long sleep can be cut short cleanly.
_stop = threading.Event()


def telegram_chat_id() -> str:
    """The chat to notify, from .env or the shell.

    Deliberately a function, not a constant: .env is loaded inside main(),
    which runs long after this module is imported, so a module-level
    os.environ.get() here would always read an empty value.
    """
    return os.environ.get("TELEGRAM_CHAT_ID", "").strip()



# --------------------------------------------------------------------------- #
# One cycle
# --------------------------------------------------------------------------- #

def run_cycle(job_filter: JobFilter, history: JobHistory) -> int:
    """Run the pipeline once. Returns how many messages were sent."""
    LOG.info("Searching %d site(s) for %d term(s)...",
             len(SITES), len(SEARCH_TERMS))
    listings = search_jobs(SITES, SEARCH_TERMS, limit=LISTINGS_PER_CYCLE)
    LOG.info("Found %d listing(s).", len(listings))
    if not listings:
        return 0

    # Skip anything already sent *before* paying for an assessment, and
    # collapse listings that are the same posting reached by different
    # search terms or found on more than one site.
    fresh, seen_here = [], set()
    old = duplicates = 0
    for listing in listings:
        identifier = _listing_id(listing)
        if identifier is None:
            fresh.append(listing)      # no usable ID; judge it later
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

    # remember=False: record only once Telegram has actually accepted it.
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

        # History is only written after a successful send, so a second copy
        # of the same posting in this batch would otherwise slip through.
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
            # Telegram may already have posted it. Record it anyway: one
            # job quietly missed beats the same job arriving twice.
            LOG.warning("Unsure whether %s was sent (%s); recording it as "
                        "sent so it cannot be posted twice.", title, exc)
            if identifier:
                history.add(identifier)
                handled.add(identifier)
            continue
        except Exception as exc:
            # Not recorded, so it will be retried next cycle.
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
    """ID for a raw JobListing or an assessed entry; None if underivable.

    Both shapes route through job_filtering.job_id, which hashes the
    listing's company, content and location — never the assessment — so the
    pre-Gemini and post-Gemini checks always agree on identity.
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


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #

def load_env() -> None:
    """Copy .env into the environment, if there is one.

    Python does not read .env by itself, and the client modules only look at
    os.environ, so without this step the file is inert.
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

    # override=False: a variable already exported in the shell wins.
    load_dotenv(ENV_FILE, override=False)
    LOG.info("Loaded %s", ENV_FILE)


def check_configuration() -> list[str]:
    """Catch missing pieces now, not 24 hours from now."""
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
    LOG.info("Received %s; finishing up.", signal.Signals(signum).name)
    _stop.set()


def main() -> int:
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
            # One bad cycle shouldn't end the run; try again next time.
            LOG.exception("Cycle %d failed.", cycle)

        if RUN_ONCE or _stop.is_set():
            break

        seconds = INTERVAL_HOURS * 3600
        next_run = datetime.now() + timedelta(seconds=seconds)
        LOG.info("Sleeping until %s.", next_run.strftime("%Y-%m-%d %H:%M"))
        if _stop.wait(seconds):  # returns early if a signal arrives
            break

    LOG.info("Stopped after %d cycle(s).", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())