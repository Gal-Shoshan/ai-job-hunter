"""Web search functionality for the job bot.

Public entry point:

    search_jobs(sites, terms, limit) -> list[JobListing]

Each site is described by a search-URL template containing ``{query}`` (and
optionally ``{page}``). Listings are extracted from schema.org JobPosting
JSON-LD when present, falling back to a link heuristic. Sites with unusual
markup can supply their own parser.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import parse_qsl, quote_plus, unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)

USER_AGENT = "JobSearchBot/0.1 (+you@example.com)"
REQUEST_TIMEOUT = 15
CRAWL_DELAY = 1.0  # seconds between requests to the same host


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class JobListing:
    title: str
    company: str | None = None
    location: str | None = None
    url: str | None = None
    posted: str | None = None
    salary: str | None = None
    description: str | None = None
    source: str | None = None       # which site it came from
    search_term: str | None = None  # which term surfaced it

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Parser = Callable[[str, str], list[JobListing]]  # (html, page_url) -> listings


class LoginWallError(RuntimeError):
    """Raised when a page turns out to require signing in.

    This bot is anonymous by design: it sends no credentials and no cookies
    you had to log in to obtain. If a site only shows listings to signed-in
    users, that site is skipped rather than worked around.
    """


@dataclass
class SiteConfig:
    """How to search one site.

    search_url: template, e.g. "https://example.com/jobs?q={query}&page={page}"
    parser:     optional site-specific parser; defaults to the generic one.
    headers:    extra request headers such as Accept-Language. Not a place
                for cookies, tokens or Authorization — see LoginWallError.
    """
    name: str
    search_url: str
    parser: Parser | None = None
    respect_robots: bool = True
    headers: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def search_jobs(
    sites: Sequence[SiteConfig | str],
    terms: Sequence[str] | str,
    limit: int = 25,
    *,
    max_pages: int = 1,
    session: requests.Session | None = None,
) -> list[JobListing]:
    """Search `sites` for `terms` and return up to `limit` job listings.

    Results are de-duplicated by URL and collected round-robin across sites so
    one busy board doesn't consume the whole quota.
    """
    if isinstance(terms, str):
        terms = [terms]
    if limit <= 0:
        return []

    configs = [_coerce_site(s) for s in sites]
    session = session or _build_session()

    results: list[JobListing] = []
    seen: set[str] = set()

    # (site, term, page) work items, ordered so sites alternate.
    for page in range(1, max_pages + 1):
        for term in terms:
            for site in configs:
                if len(results) >= limit:
                    return results[:limit]
                try:
                    found = _search_one(session, site, term, page)
                except Exception as exc:  # network, parse, anything
                    LOG.warning("%s failed for %r (page %d): %s",
                                site.name, term, page, exc)
                    continue

                for job in found:
                    key = job.url or f"{job.source}:{job.title}:{job.company}"
                    if key in seen:
                        continue
                    seen.add(key)
                    job.search_term = term
                    results.append(job)
                    if len(results) >= limit:
                        return results[:limit]

    return results[:limit]


def _search_one(
    session: requests.Session,
    site: SiteConfig,
    term: str,
    page: int,
) -> list[JobListing]:
    url = site.search_url.format(query=quote_plus(term), page=page)

    if site.respect_robots and not _robots_allows(session, url):
        LOG.info("robots.txt disallows %s", url)
        return []

    html = _fetch(session, url, site.headers)
    parser = site.parser or extract_listings
    listings = parser(html, url)
    for job in listings:
        job.source = job.source or site.name
    return listings


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract_listings(html: str, page_url: str) -> list[JobListing]:
    """Generic extractor: structured data first, link heuristic as fallback."""
    soup = BeautifulSoup(html, "html.parser")
    listings = _from_json_ld(soup, page_url)
    if not listings:
        listings = _from_links(soup, page_url)
    return listings


def _from_json_ld(soup: BeautifulSoup, page_url: str) -> list[JobListing]:
    listings: list[JobListing] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for node in _walk(data):
            if not isinstance(node, dict):
                continue
            node_type = node.get("@type")
            types = node_type if isinstance(node_type, list) else [node_type]
            if "JobPosting" not in types:
                continue
            listings.append(_job_from_node(node, page_url))
    return listings


def _walk(node: Any) -> Iterable[Any]:
    """Yield every dict/list nested anywhere inside a JSON-LD blob."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _job_from_node(node: dict[str, Any], page_url: str) -> JobListing:
    url = node.get("url") or node.get("sameAs") or ""
    return JobListing(
        title=_text(node.get("title")) or "(untitled)",
        company=_text(_get(node, "hiringOrganization", "name")),
        location=_location(node),
        url=urljoin(page_url, url) if url else None,
        posted=_text(node.get("datePosted")),
        salary=_salary(node.get("baseSalary")),
        description=_strip_html(node.get("description")),
    )


def _location(node: dict[str, Any]) -> str | None:
    if str(node.get("jobLocationType", "")).upper() == "TELECOMMUTE":
        return "Remote"
    loc = node.get("jobLocation")
    if isinstance(loc, list):
        loc = loc[0] if loc else None
    address = _get(loc, "address") if isinstance(loc, dict) else None
    if isinstance(address, str):
        return address
    if not isinstance(address, dict):
        return None
    parts = [
        address.get("addressLocality"),
        address.get("addressRegion"),
        address.get("addressCountry") if isinstance(
            address.get("addressCountry"), str) else
        _get(address, "addressCountry", "name"),
    ]
    joined = ", ".join(p for p in parts if isinstance(p, str) and p.strip())
    return joined or None


def _salary(node: Any) -> str | None:
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return None
    currency = node.get("currency") or ""
    value = node.get("value")
    if isinstance(value, dict):
        lo, hi = value.get("minValue"), value.get("maxValue")
        unit = value.get("unitText", "")
        amount = (f"{lo}-{hi}" if lo and hi else str(lo or hi or "")).strip()
        if not amount:
            return None
        return " ".join(p for p in (currency, amount, unit) if p).strip()
    if value:
        return f"{currency} {value}".strip()
    return None


# --- fallback link heuristics ---------------------------------------------- #
#
# A job board's search page is mostly navigation. Matching any href containing
# "/job" pulls in the menu ("דרושים הייטק"), category pages and pagination, so
# a link is only accepted when its URL is shaped like a single posting AND it
# sits outside the page's chrome.

# Path segments that introduce one posting.
JOB_SEGMENTS = {
    "job", "jobs", "jobad", "joboffer", "job-details", "jobdetails",
    "position", "positions", "vacancy", "vacancies", "opening", "openings",
    "career", "careers", "listing", "listings", "remote-jobs",
    "משרה", "משרות", "drushim",
}

# Segments that mean "a page of many jobs", never one posting.
INDEX_SEGMENTS = {
    "search", "searches", "results", "category", "categories", "cat",
    "browse", "tag", "tags", "topic", "topics", "area", "areas", "city",
    "cities", "region", "regions", "field", "fields", "filter", "filters",
    "index", "all", "list", "sitemap", "page", "pages", "login", "register",
    "signup", "about", "contact", "faq", "blog", "article", "articles",
    "news", "companies", "employers", "advertise", "terms", "privacy",
    "relocation", "salary", "salaries", "tips", "guide", "guides", "home",
    "קטגוריה", "חיפוש", "רילוקיישן",
}

# Query keys that carry a posting's id (AllJobs uses ?JobID=123456).
ID_QUERY_KEYS = {
    "id", "jobid", "job_id", "jid", "jk", "gh_jid", "positionid",
    "position_id", "vacancyid", "vacancy_id", "oid", "pid", "jobcode",
    "jobnumber", "reqid", "requisitionid",
}

# Query keys that mark a search or category page.
INDEX_QUERY_KEYS = {
    "q", "query", "search", "keyword", "keywords", "term", "terms", "page",
    "category", "cat", "field", "area", "region", "city", "sort", "filter",
}

# Containers whose links are site chrome, not results.
CHROME_TAGS = {"nav", "header", "footer", "aside"}
CHROME_HINT = re.compile(
    r"nav|menu|header|footer|breadcrumb|sidebar|side-bar|tabs?\b|filter|"
    r"pagination|pager|cookie|banner|social|lang|skip", re.I)

# Link text that names a section rather than a role.
GENERIC_TEXT = re.compile(
    r"^(all|more|view|see|browse|show|next|prev|previous|back|home|jobs|"
    r"careers|vacancies|positions|login|register|sign in|sign up|apply)\b"
    r"|^(כל|עוד|חיפוש|דרושים|משרות)\s",
    re.I,
)

_DIGITS = re.compile(r"\d{3,}")


def _from_links(soup: BeautifulSoup, page_url: str) -> list[JobListing]:
    """Fallback: anchors whose URL is shaped like a single job posting."""
    base_host = _host(page_url)
    out: list[JobListing] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        if _in_chrome(anchor):
            continue

        title = " ".join(anchor.get_text(" ", strip=True).split())
        if not _plausible_title(title):
            continue

        absolute = urljoin(page_url, anchor["href"]).split("#")[0]
        if not _is_posting_url(absolute, base_host):
            continue
        if absolute in seen:
            continue

        seen.add(absolute)
        out.append(JobListing(title=title, url=absolute))

    return out


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _in_chrome(tag) -> bool:
    """True if the link sits in a nav, header, footer or similar wrapper."""
    for parent in tag.parents:
        name = getattr(parent, "name", None)
        if name is None:
            continue
        if name in CHROME_TAGS:
            return True
        attrs = getattr(parent, "attrs", {}) or {}
        classes = attrs.get("class") or []
        if isinstance(classes, str):
            classes = [classes]
        blob = " ".join([*classes, str(attrs.get("id", "")),
                         str(attrs.get("role", ""))]).strip()
        if blob and CHROME_HINT.search(blob):
            return True
    return False


def _plausible_title(title: str) -> bool:
    if not (5 <= len(title) <= 120):
        return False
    if GENERIC_TEXT.search(title):
        return False
    return title.count(" ") <= 20      # a paragraph, not a job title


def _is_posting_url(url: str, base_host: str) -> bool:
    """Accept only URLs shaped like one posting, not an index or category."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if base_host and host and host != base_host \
            and not host.endswith("." + base_host):
        return False               # an advert or an off-site link

    segments = [unquote(seg).lower() for seg in parsed.path.split("/") if seg]
    if not segments:
        return False

    query = {k.lower(): v for k, v in parse_qsl(parsed.query)}
    has_index_query = any(k in INDEX_QUERY_KEYS for k in query)

    # An explicit job-id parameter is decisive: /Search/Single.aspx?JobID=12345
    if not has_index_query:
        for key, value in query.items():
            if key in ID_QUERY_KEYS and _DIGITS.search(value) \
                    and any(word in key for word in ("job", "vacancy", "position")):
                return True

    if any(seg in INDEX_SEGMENTS for seg in segments):
        return False
    if has_index_query:
        return False

    job_at = next((i for i, seg in enumerate(segments)
                   if seg in JOB_SEGMENTS), None)
    if job_at is None:
        return False

    tail = segments[job_at + 1:]
    if not tail:
        return False               # "/jobs/" is the index itself

    # A numeric id anywhere after the job segment: /jobs/12345, /job/12345-dev
    if any(_DIGITS.search(seg) for seg in tail):
        return True
    for key, value in query.items():
        if key in ID_QUERY_KEYS and _DIGITS.search(value):
            return True

    # No id: accept only a specific multi-word slug, as used by boards like
    # We Work Remotely (/remote-jobs/acme-corp-backend-engineer).
    last = tail[-1]
    return last.count("-") >= 2 and len(last) >= 12


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

_last_request_at: dict[str, float] = {}
_robots_cache: dict[str, RobotFileParser | None] = {}


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def _fetch(session: requests.Session, url: str,
           headers: dict[str, str] | None = None) -> str:
    host = urlparse(url).netloc
    elapsed = time.monotonic() - _last_request_at.get(host, 0.0)
    if elapsed < CRAWL_DELAY:
        time.sleep(CRAWL_DELAY - elapsed)

    extra = headers or {}
    blocked = {"authorization", "cookie"} & {k.lower() for k in extra}
    if blocked:
        raise ValueError(
            f"Refusing to send credential headers {sorted(blocked)}: this bot "
            "only reads publicly visible listings."
        )

    LOG.debug("GET %s", url)
    response = session.get(url, headers=extra, timeout=REQUEST_TIMEOUT)
    _last_request_at[host] = time.monotonic()

    if response.status_code in (401, 403):
        raise LoginWallError(
            f"{url} returned {response.status_code} — listings are not public."
        )
    response.raise_for_status()
    if _is_login_page(response):
        raise LoginWallError(f"{url} redirected to a sign-in page.")
    return response.text


_LOGIN_PATH = re.compile(
    r"/(log[-_]?in|sign[-_]?in|sign[-_]?up|auth|session|checkpoint)\b", re.I
)


def _is_login_page(response: requests.Response) -> bool:
    """True if we landed on a sign-in wall rather than search results."""
    if _LOGIN_PATH.search(urlparse(response.url).path):
        return True
    head = response.text[:20000].lower()
    return 'type="password"' in head or "type='password'" in head


def _robots_allows(session: requests.Session, url: str) -> bool:
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root not in _robots_cache:
        parser: RobotFileParser | None = RobotFileParser()
        try:
            resp = session.get(f"{root}/robots.txt", timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                parser = None  # no usable robots.txt -> treat as allowed
            else:
                parser.parse(resp.text.splitlines())
        except requests.RequestException:
            parser = None
        _robots_cache[root] = parser
    parser = _robots_cache[root]
    return True if parser is None else parser.can_fetch(USER_AGENT, url)


def _coerce_site(site: SiteConfig | str) -> SiteConfig:
    if isinstance(site, SiteConfig):
        return site
    if "{query}" not in site:
        raise ValueError(
            f"Site {site!r} must be a SiteConfig or a URL template "
            "containing '{query}'."
        )
    return SiteConfig(name=urlparse(site).netloc or site, search_url=site)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def _get(node: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = " ".join(value.split())
        return cleaned or None
    return None


def _strip_html(value: Any, max_chars: int = 600) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(BeautifulSoup(value, "html.parser").get_text(" ").split())
    if not text:
        return None
    return text[:max_chars].rstrip() + "…" if len(text) > max_chars else text