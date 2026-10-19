"""Stage 1: extraction. Incremental, idempotent, checkpointed.

The HTTP layer is a small `Transport` interface so you can swap the curl+cookie approach for
Reddit's official OAuth API (e.g. via PRAW) without touching the rest of the pipeline.
"""
from __future__ import annotations

import json
import logging
import random
import subprocess
import time
from typing import Any, Protocol

from pydantic import ValidationError

from .config import Settings
from .models import RawPost
from .storage import Database, now

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


class BlockedError(FetchError):
    """401/403 from Reddit itself: retrying won't help (cookie/UA missing or expired)."""


class SubredditUnavailable(FetchError):
    """This one community can't be read (private, banned, quarantined, not found). Others are fine."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def unavailable_reason(body: str, status: str) -> str | None:
    """Tell 'this subreddit is private/banned' apart from 'Reddit is blocking us'."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and data.get("reason") in ("private", "banned", "quarantined", "gated"):
        return str(data["reason"])
    return "not_found" if status == "404" else None


class Transport(Protocol):
    def get_json(self, url: str) -> Any: ...


class CurlTransport:
    """The curl + User-Agent + cookie combination from the original scraper.

    The cookie is passed on stdin (`-H @-`) so it never shows up in `ps` output.
    """

    def __init__(self, user_agent: str, cookie: str, retries: int = 4, timeout: int = 30):
        self.user_agent, self.cookie, self.retries, self.timeout = user_agent, cookie, retries, timeout

    def _run(self, url: str) -> tuple[str, str]:
        cmd = ["curl", "-sS", "-L", "--compressed", "--max-time", str(self.timeout), "-A", self.user_agent]
        stdin = None
        if self.cookie:
            cmd += ["-H", "@-"]
            stdin = f"Cookie: {self.cookie}\n"
        cmd += ["-w", "\n%{http_code}", url]
        proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
        if proc.returncode != 0:
            raise FetchError(proc.stderr.strip() or f"curl exited with {proc.returncode}")
        body, _, status = proc.stdout.rpartition("\n")
        return body, status

    def get_json(self, url: str) -> Any:
        last = "unknown error"
        for attempt in range(1, self.retries + 1):
            try:
                body, status = self._run(url)
            except FetchError as exc:
                last = str(exc)
            else:
                if status == "200" and not body.lstrip().startswith("<"):
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError:
                        last = "invalid JSON body"
                elif status in ("401", "403", "404"):
                    reason = unavailable_reason(body, status)
                    if reason:
                        raise SubredditUnavailable(reason)
                    raise BlockedError(f"HTTP {status}: blocked (check REDDIT_COOKIE / user agent)")
                elif status == "429":
                    wait = 30 * attempt
                    log.warning("429 rate limited, sleeping %ss", wait)
                    last = "HTTP 429 rate limited"
                    time.sleep(wait)
                    continue
                else:
                    last = f"HTTP {status} (non-JSON body, likely blocked)" if status == "200" else f"HTTP {status}"
            time.sleep(min(30, 2 ** attempt) + random.random())
        raise FetchError(f"{url}: {last}")


def listing_url(subreddit: str, after: str | None) -> str:
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=100&raw_json=1"
    return url + (f"&after={after}" if after else "")


def fetch_posts(db: Database, transport: Transport, s: Settings, subreddit: str, backfill: bool = False) -> dict:
    """Walk one subreddit's /new and upsert every post into the raw store.

    incremental (default): start at the newest page, stop at the first page that reaches a post
                           we've already seen (checkpoint: newest created_utc per subreddit).
    backfill:              walk as deep as Reddit allows, resuming from a saved page cursor.
    Both stop early at `max_age_days`. Re-running is always safe: raw posts are upserted by Reddit ID.
    """
    sub = subreddit.lower()
    latest_key, cursor_key = f"{sub}:latest_created_utc", f"{sub}:backfill_after"
    latest = float(db.get_state(latest_key, "0") or 0)
    after = db.get_state(cursor_key) if backfill else None
    stop_at = 0.0 if backfill else latest
    too_old = time.time() - s.max_age_days * 86400 if s.max_age_days else 0.0

    stats = {"pages": 0, "new": 0, "updated": 0, "invalid": 0}
    newest = latest
    for _ in range(s.max_pages):
        data = transport.get_json(listing_url(sub, after))
        try:
            children, after = data["data"]["children"], data["data"].get("after")
        except (KeyError, TypeError) as exc:
            raise FetchError(f"unexpected listing shape: {exc}") from exc

        reached_known = False
        for child in children:
            payload = child.get("data", {})
            try:
                post = RawPost.model_validate(payload)   # schema check at the boundary
            except ValidationError as exc:
                stats["invalid"] += 1
                log.warning("invalid post skipped: %s", str(exc).splitlines()[0])
                continue
            if too_old and post.created_utc < too_old:
                reached_known = True            # past the age window: nothing older is wanted
                continue
            stats["new" if db.upsert_raw_post(post.id, sub, payload) else "updated"] += 1
            newest = max(newest, post.created_utc)
            if not backfill and post.created_utc <= stop_at:
                reached_known = True

        stats["pages"] += 1
        db.set_state(latest_key, str(newest))
        log.info("page %d: %d posts (after=%s) new=%d", stats["pages"], len(children), after, stats["new"])
        if backfill and after:
            db.set_state(cursor_key, after)
        if not after or reached_known:
            break
        time.sleep(s.fetch_delay)

    if backfill and not after:
        db.delete_state(cursor_key)
    return stats


def fetch_all(db: Database, transport: Transport, s: Settings, backfill: bool = False) -> dict:
    """Fetch every approved subreddit (or just `s.subreddit`), isolating per-community failures.

    A private/banned community is marked and skipped; a Reddit-level block stops the whole run.
    """
    if s.subreddit:
        db.upsert_subreddit(s.subreddit, via="manual", status="approved", decided_by="manual", force_status=True)
        targets = [db.get_subreddit(s.subreddit)]
    else:
        targets = db.approved_subreddits()
    if s.max_subreddits:
        targets = targets[: s.max_subreddits]
    if not targets:
        log.warning("No approved subreddits. Run discover -> classify -> review-export/review-apply first, "
                    "or pass --subreddit NAME.")

    results: dict[str, dict] = {}
    for i, row in enumerate(targets):
        name = row["name"]
        try:
            results[name] = fetch_posts(db, transport, s, name, backfill=backfill)
            db.update_subreddit(name, last_fetched_at=now(), last_error=None)
        except SubredditUnavailable as exc:
            log.warning("r/%s unavailable (%s); skipping", name, exc.reason)
            db.update_subreddit(name, last_error=exc.reason, last_fetched_at=now())
            results[name] = {"error": exc.reason}
        except BlockedError as exc:
            log.error("%s - stopping fetch", exc)
            results[name] = {"error": "blocked"}
            break
        except FetchError as exc:
            log.warning("r/%s failed: %s", name, exc)
            db.update_subreddit(name, last_error=str(exc)[:200])
            results[name] = {"error": str(exc)[:200]}
        if i + 1 < len(targets):
            time.sleep(s.fetch_delay)
    return {"subreddits": len(results), "new_posts": sum(r.get("new", 0) for r in results.values()),
            "errors": {k: v["error"] for k, v in results.items() if "error" in v}}


def comments_url(post_id: str) -> str:
    return f"https://www.reddit.com/comments/{post_id}.json?limit=100&depth=2&sort=top&raw_json=1"


def fetch_comments(db: Database, transport: Transport, s: Settings, limit: int = 0) -> dict:
    """Fetch comment threads only for posts that survived filtering (best engagement first)."""
    todo = db.posts_needing_comments(s.subreddit, limit, s.use_triage, s.triage_prompt_version)
    stats = {"todo": len(todo), "fetched": 0, "failed": 0}
    for row in todo:
        try:
            payload = transport.get_json(comments_url(row["id"]))
        except BlockedError as exc:
            log.error("%s - stopping comment fetch", exc)
            db.record_failure(row["id"], "comments", None, str(exc))
            stats["failed"] += 1
            break
        except FetchError as exc:
            db.record_failure(row["id"], "comments", None, str(exc))
            stats["failed"] += 1
            continue
        db.save_comments(row["id"], payload, "api")
        db.resolve_failure(row["id"], "comments")
        stats["fetched"] += 1
        time.sleep(s.fetch_delay)
    return stats
