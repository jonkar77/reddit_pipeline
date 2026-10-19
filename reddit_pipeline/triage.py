"""Triage: a cheap first pass over title + body only (no comments).

With many communities, fetching comments and running the full analysis for every post is the slow part.
Triage rejects obvious noise first, so comment requests and full LLM calls go only to promising posts.
It is tuned for recall: when unsure it keeps the post.
"""
from __future__ import annotations

import logging
import time
from collections import Counter

from .analysis import input_hash
from .config import Settings
from .llm import LLMClient, LLMError, structured
from .models import Triage
from .storage import Database

log = logging.getLogger(__name__)

TRIAGE_SYSTEM = """You are a fast first-pass filter for posts from trading-related Reddit communities.
From ONLY the post's subreddit, title and body, decide whether it is worth a deeper look for finding
real, concrete problems a software product could solve: missing tools, broken workflows, costly or
unreliable data, repeated manual work, tool requests, specific frustrations.

Set worth_analyzing=false for: ads and self-promotion, P&L flexing, news, memes and jokes, pure market
opinion or speculation, motivational posts, and generic beginner questions with no tooling angle.
When in doubt, set worth_analyzing=true: looking twice is cheaper than missing a real problem.
post_type: the best-fit category. reason: one short phrase."""


def triage_context(post, s: Settings) -> str:
    body = (post["selftext"] or "")[:1200] or "(no body, link/title-only post)"
    return (f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\n"
            f"FLAIR: {post['flair'] or 'none'}\nBODY:\n{body}")


def run_triage(db: Database, llm: LLMClient, s: Settings, limit: int = 0) -> dict:
    """Triage eligible posts (relative-engagement order). `limit` caps NEW model calls."""
    run_id = db.start_run("triage", llm.model, s.triage_prompt_version, {"limit": limit})
    stats: Counter = Counter()
    latencies: list[float] = []
    for p in db.eligible_posts(s.subreddit or None):
        if limit and stats["triaged"] >= limit:
            break
        ctx = triage_context(p, s)
        key = input_hash(TRIAGE_SYSTEM, ctx, llm.model, s.triage_prompt_version)
        if db.get_triage_hash(p["id"]) == key:
            stats["cached"] += 1
            continue
        t0 = time.monotonic()
        try:
            res = structured(llm, TRIAGE_SYSTEM, ctx, Triage)
        except LLMError as exc:
            db.record_failure(p["id"], "triage", key, str(exc))
            stats["failed"] += 1
            log.warning("triage failed for %s: %s", p["id"], exc)
            continue
        latencies.append(time.monotonic() - t0)
        v = res.value
        db.save_triage(p["id"], key, llm.model, s.triage_prompt_version, v.worth_analyzing, v.post_type, v.reason)
        db.resolve_failure(p["id"], "triage")
        stats["triaged"] += 1
        stats["kept" if v.worth_analyzing else "rejected"] += 1
    metrics = dict(stats)
    metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 2) if latencies else 0
    db.finish_run(run_id, metrics)
    return metrics
