"""Pre-fetch layer, step 1: find candidate communities.

Three sources feed one `subreddits` table (all land as `candidate`, except your own seeds):
  search     keyword queries against Reddit's subreddit search (paginated)
  snowball   r/xyz mentions + crosspost sources in data we already hold, and sidebar links of approved subs
  seeds      communities you name yourself (approved immediately)
Nothing here decides relevance. That is curation.py's job.
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from urllib.parse import quote_plus

from pydantic import ValidationError

from .config import Settings
from .fetcher import BlockedError, FetchError, SubredditUnavailable, Transport
from .models import RawSubreddit
from .processing import parse_comments
from .storage import Database, now

log = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "trading", "trader", "day trading", "swing trading", "options", "futures", "forex", "crypto",
    "stocks", "stock market", "investing", "quant", "algo trading", "backtesting", "prop firm",
    "technical analysis", "wallstreetbets", "securities",
]
MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])r/([A-Za-z0-9_]{3,21})(?![A-Za-z0-9_])")


def search_url(query: str, after: str | None) -> str:
    url = f"https://www.reddit.com/subreddits/search.json?q={quote_plus(query)}&limit=100&raw_json=1"
    return url + (f"&after={after}" if after else "")


def about_url(name: str) -> str:
    return f"https://www.reddit.com/r/{name}/about.json?raw_json=1"


def meta_from(raw: RawSubreddit) -> dict:
    """Flatten a validated payload into `subreddits` columns."""
    desc = "\n\n".join(x for x in (raw.public_description, raw.description) if x)
    return {"display_name": raw.display_name, "title": raw.title or "", "description": desc,
            "subscribers": raw.subscribers, "created_utc": raw.created_utc,
            "over18": int(bool(raw.over18)), "subreddit_type": raw.subreddit_type or "public",
            "quarantine": int(bool(raw.quarantine))}


def parse_listing(data) -> tuple[list[RawSubreddit], str | None]:
    try:
        children, after = data["data"]["children"], data["data"].get("after")
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected subreddit listing shape: {exc}") from exc
    out: list[RawSubreddit] = []
    for child in children:
        if child.get("kind") != "t5":
            continue
        try:
            out.append(RawSubreddit.model_validate(child.get("data", {})))
        except ValidationError:
            continue                                   # schema drift: skip, don't crash discovery
    return out, after


def search_subreddits(transport: Transport, query: str, pages: int, delay: float = 1.0) -> list[RawSubreddit]:
    found: list[RawSubreddit] = []
    after = None
    for _ in range(pages):
        batch, after = parse_listing(transport.get_json(search_url(query, after)))
        found.extend(batch)
        if not after:
            break
        time.sleep(delay)
    return found


def fetch_about(transport: Transport, name: str) -> RawSubreddit:
    data = transport.get_json(about_url(name))
    try:
        return RawSubreddit.model_validate(data["data"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise FetchError(f"unexpected about payload for r/{name}: {exc}") from exc


# ---- snowballing --------------------------------------------------------------------
def extract_mentions(text: str | None) -> set[str]:
    return {m.lower() for m in MENTION_RE.findall(text or "")}


def collect_mentions(db: Database) -> Counter:
    """Count how many THREADS mention each community (post text, its comments, crosspost sources)."""
    per_thread: dict[str, set[str]] = {}
    for post_id, payload in db.iter_raw_posts():
        names = extract_mentions(f"{payload.get('title', '')}\n{payload.get('selftext', '')}")
        for parent in payload.get("crosspost_parent_list") or []:
            if parent.get("subreddit"):
                names.add(parent["subreddit"].lower())
        per_thread[post_id] = names
    for post_id, payload, fmt in db.iter_raw_comments():
        for c in parse_comments(payload, fmt):
            per_thread.setdefault(post_id, set()).update(extract_mentions(c.get("body", "")))
    counts: Counter = Counter()
    for names in per_thread.values():
        counts.update(names)
    return counts


def snowball(db: Database, s: Settings) -> dict:
    stats = {"new_from_mentions": 0, "new_from_sidebars": 0}
    for name, count in collect_mentions(db).items():
        if count >= s.snowball_min_mentions and db.upsert_subreddit(name, via="mention", mentions=count):
            stats["new_from_mentions"] += 1
    for row in db.approved_subreddits():               # communities you already trust recommend others
        for name in extract_mentions(row["description"]):
            if name != row["name"] and db.upsert_subreddit(name, via=f"sidebar:{row['name']}"):
                stats["new_from_sidebars"] += 1
    return stats


def hydrate(db: Database, transport: Transport, s: Settings, limit: int = 50) -> int:
    """Fetch metadata for stub candidates found by snowballing (no description/subscriber data yet)."""
    done = 0
    for row in db.subreddits_needing_metadata(limit):
        try:
            raw = fetch_about(transport, row["name"])
            db.upsert_subreddit(row["name"], meta=meta_from(raw))
        except SubredditUnavailable as exc:
            db.update_subreddit(row["name"], status="rejected", decided_by="rule", reason=exc.reason,
                                metadata_at=now(), classified_at=now())
        except BlockedError:
            raise
        except FetchError as exc:
            log.warning("hydrate r/%s failed: %s", row["name"], exc)
            continue
        done += 1
        time.sleep(s.fetch_delay)
    return done


def run_discovery(db: Database, transport: Transport, s: Settings, *, queries: list[str] | None = None,
                  seeds: list[str] | None = None, do_snowball: bool = True, hydrate_limit: int = 50) -> dict:
    stats: Counter = Counter()
    for q in queries or DEFAULT_QUERIES:
        try:
            found = search_subreddits(transport, q, s.search_pages, s.fetch_delay)
        except BlockedError:
            raise
        except FetchError as exc:
            log.warning("search %r failed: %s", q, exc)
            stats["search_failed"] += 1
            continue
        stats["search_hits"] += len(found)
        for raw in found:
            if db.upsert_subreddit(raw.display_name, meta=meta_from(raw), via=f"search:{q}"):
                stats["new_from_search"] += 1
        log.info("search %r: %d communities", q, len(found))
        time.sleep(s.fetch_delay)

    for name in seeds or []:
        name = name.strip().removeprefix("r/").strip()
        if not name:
            continue
        try:
            raw = fetch_about(transport, name)
            db.upsert_subreddit(name, meta=meta_from(raw), via="seed", status="approved",
                                decided_by="seed", force_status=True)
            stats["seeds_added"] += 1
        except SubredditUnavailable as exc:
            log.warning("seed r/%s unavailable: %s", name, exc.reason)
            stats["seeds_unavailable"] += 1

    if do_snowball:
        stats.update(snowball(db, s))
    stats["hydrated"] = hydrate(db, transport, s, hydrate_limit)
    return dict(stats)
