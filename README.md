# AI Job Hunter

An autonomous agent that scrapes public Israeli tech job boards, scores every
listing against your CV with Gemini, filters the results against rules you
declare, and delivers the survivors to Telegram — once a day, unattended.

Job boards optimise for recruiters, not candidates. A search for "junior C++"
returns hundreds of roles demanding five years of experience, and the handful
worth applying to are buried. This agent inverts that: it reads the adverts so
you don't have to, judges each one against your actual résumé, and messages you
only when something genuinely fits.

## Key Features

- **CV-aware scoring.** Your résumé is sent as an inline PDF with every
  listing, so matches are judged against real experience rather than keywords.
- **Declarative filtering.** Compatibility, match score, commute, office days
  and salary thresholds live in `job_filtering.json`, editable without touching
  code. Bounds can depend on other fields — the salary floor rises with the
  number of office days.
- **Never sends the same job twice.** Listings are identified by a hash of the
  advert itself, so a posting keeps one identity whether it is seen before or
  after the model judges it, and across re-postings with cosmetic edits.
- **Anonymous, polite scraping.** No credentials, no cookies, `robots.txt`
  honoured, per-host rate limiting. Sites that hide listings behind a login are
  skipped rather than worked around.
- **Quota-aware Gemini calls.** Requests are paced to the free-tier limit and
  retried on quota and overload errors, which say nothing about the listing.
- **Duplicate-safe delivery.** A Telegram send whose outcome is unknown is
  never retried, because the Bot API has no idempotency key.
- **Interactive debugging mode.** `tests.py` runs the identical pipeline but
  pauses at every module boundary so you can inspect what each stage returned.

## Technical Architecture & Stack

**Stack:** Python 3.11+ · `requests` · `BeautifulSoup4` · `google-genai`
(Gemini 3.5 Flash-Lite) · Telegram Bot API · `python-dotenv` (optional)

**Pipeline.** Each cycle runs search → deduplicate → assess → filter → notify.
Deduplication happens *before* assessment, so no quota is spent on a listing
that has already been sent.

**Scraping.** Sites are described by URL templates. `{query}` is substituted
with a search term and `{page}` with a page number; a template without
`{query}` ignores the terms and is fetched once per cycle rather than once per
term. Extraction tries schema.org JobPosting JSON-LD first, then a JSON API
parser, then a link heuristic that accepts only URLs shaped like a single
posting and sitting outside the page's navigation. Sites with awkward markup
supply their own parser, keyed on stable URL patterns rather than CSS classes.

**Fair collection.** Results are collected into one bucket per configured URL,
merged round-robin into a bucket per site, then merged round-robin again into
the final result. A busy board cannot crowd out a quiet one, and a single URL
cannot spend its site's whole quota. `limit` is a per-site quota, not a total.

**Assessment.** `llm_format_answer.json` is the single source of truth for the
model's output: a Gemini response schema is derived from it, replies are
validated against it, and fields whose description says `never 0` are re-asked
when the model declines to estimate. Calls are paced to `REQUESTS_PER_MINUTE`,
and 429/5xx responses are waited out using the API's own stated retry delay.

**Error handling.** Failures are classified by whether repeating them is safe.
Quota and overload errors are retried; malformed-output errors are re-asked
once; configuration errors abort the batch rather than logging the same warning
per listing. On the Telegram side, only a failed connection and an explicit 429
are retried — a read timeout or 5xx raises `TelegramUncertain`, and the caller
records the job as sent so it cannot arrive twice.

**Session management.** HTTP sessions are created once and reused. The résumé
is cached on `(path, mtime)`, so a long run reads the PDF once. The Gemini
client is built lazily, so importing a module never requires a key. History is
written atomically, so a crash mid-write cannot corrupt it.

## Project Structure

```
ai-job-hunter/
├── agent.py                 Entry point: the 24-hour loop, config, signals
├── tests.py                 Same pipeline, paused at every module boundary
│
├── job_search.py            Scraping: SiteConfig, extractors, per-site quotas
├── gemini_client.py         Assessment: prompting, schema, pacing, retries
├── job_filtering.py         JobFilter rules + JobHistory deduplication
├── telegram_client.py       HTML rendering, splitting, duplicate-safe sending
│
├── llm_format_answer.json   Answer template — drives schema and validation
├── job_filtering.json       Filter rules: thresholds and conditional bounds
├── job_history.json         Generated: ids already delivered
├── resume.pdf               Your CV (not in version control)
└── .env                     Your secrets (not in version control)
```

## Setup & Quickstart

**1. Install dependencies**

```bash
pip install requests beautifulsoup4 google-genai python-dotenv
```

**2. Provide credentials**

Create a `.env` file beside `agent.py`, which contains the following variables:

```
GEMINI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

| Variable | Required | How to get it |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | [Google AI Studio](https://aistudio.google.com/apikey) |
| `TELEGRAM_BOT_TOKEN` | Unless `DRY_RUN` | Message [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Unless `DRY_RUN` | Message your bot, then read `https://api.telegram.org/bot<TOKEN>/getUpdates` |
| `JOB_TEMPLATE` | No | Path to the answer template. Defaults to `llm_format_answer.json` |

**3. Add your CV**

Drop `resume.pdf` beside `agent.py`.

**4. Tune the search**

Edit the Parameters block at the top of `agent.py`: `SEARCH_TERMS`,
`JOB_FIELD`, `HOME_LOCATION` (the origin for commute estimates), and
`LISTINGS_PER_CYCLE` (a per-site quota — two sites at 70 means up to 140
assessments per cycle).

**5. Run it**

```bash
python agent.py
```

The agent searches immediately, then sleeps until the next cycle. `Ctrl-C`
stops it cleanly at the end of the current cycle.

### Tuning notes

- **Filter too tight?** Every drop is logged with its reason. If
  `salary_min_ils_monthly` appears often, the floor in `job_filtering.json` is
  the thing to lower — note that `default` applies whenever `days_per_week` is
  unknown, which is most of the time.
- **Search terms not all used?** A single results page can fill a site's quota,
  so later terms are skipped. Raise `LISTINGS_PER_CYCLE` or use fewer terms.
- **Rate limits.** Check your real quota at
  [AI Studio](https://aistudio.google.com/rate-limit) and set
  `REQUESTS_PER_MINUTE` in `gemini_client.py` to match.

### Adding a site

Append a `SiteConfig` to `SITES`. Anything with a free-text search parameter
works out of the box:

```python
SiteConfig(name="Example", search_url="https://example.com/jobs?q={query}&page={page}")
```

Entries sharing a `name` pool into one bucket and split that site's quota.
Supply `parser=` only when the generic extractors come back empty.