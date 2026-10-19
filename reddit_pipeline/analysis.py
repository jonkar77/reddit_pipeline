"""Stage 3: local LLM extraction. Cached by input hash, validated, failures go to a replayable queue."""
from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter

from .config import Settings
from .llm import LLMClient, LLMError, structured
from .models import Extraction
from .processing import build_context, parse_comments
from .storage import Database

log = logging.getLogger(__name__)

EXTRACT_SYSTEM = """You analyze posts (with their top comments) from trading-related Reddit communities; the
subreddit is named in each post.
Goal: find REAL, concrete problems the community has that a software product could solve.

NOISE - set has_pain=false for these:
- ads, self-promotion, product/course/signal-service plugs, "check out my bot/repo"
- flexing: P&L screenshots, "I made $X", bragging about returns
- news, memes, jokes, vague market opinions, motivational posts
- beginner "where do I start" questions with no unmet tooling need

REAL PAIN - set has_pain=true only when the author or commenters describe a specific
frustration, time sink, cost, reliability issue, missing tool, or broken workflow.

Rules:
- Use only what is in the text. Never invent problems.
- intensity: 1 mild annoyance .. 5 blocking/costly/repeated by many people. Use 1 when no pain.
- Comments that confirm "I have this problem too" or describe workarounds raise the intensity.
- software_opportunity: one sentence on what could be built. Empty string if nothing.
- willingness_to_pay (pick the strongest evidence IN THE TEXT, otherwise "none"):
    explicit_price       a concrete price/cost is stated (paid or would pay)
    paid_tool_mentioned  a paid tool/service is named as used or tried for this problem
    would_pay            someone says they would pay / hire for a solution
    none                 no payment evidence
- evidence: at most 2 SHORT verbatim quotes (under 25 words each)."""


def input_hash(system: str, context: str, model: str, prompt_version: str) -> str:
    """Cache key. Changes if the post/comments, prompt, prompt version or model changes."""
    return hashlib.sha256("\x1f".join([prompt_version, model, system, context]).encode()).hexdigest()


def run_analysis(db: Database, llm: LLMClient, s: Settings, *, limit: int = 0,
                 use_comments: bool = True, only_failed: bool = False) -> dict:
    """Analyse eligible posts, best engagement first. `limit` caps NEW model calls (cache hits are free)."""
    system = EXTRACT_SYSTEM
    posts = db.eligible_posts(s.subreddit or None)
    triage = db.triage_map(s.triage_prompt_version) if s.use_triage else None
    if only_failed:
        failed = {f["post_id"] for f in db.unresolved_failures("analysis")}
        posts = [p for p in posts if p["id"] in failed]

    run_id = db.start_run("analysis", llm.model, s.prompt_version, {"limit": limit, "use_comments": use_comments})
    stats: Counter = Counter()
    latencies: list[float] = []
    for p in posts:
        if limit and stats["analyzed"] >= limit:
            break
        if triage is not None:                      # triage gate: only promising posts get the full analysis
            verdict = triage.get(p["id"])
            if verdict is None:
                stats["skipped_not_triaged"] += 1
                continue
            if not verdict:
                stats["skipped_triage_rejected"] += 1
                continue
        comments = None
        if use_comments:
            stored = db.get_comments(p["id"])
            if stored is None and p["num_comments"] > 0:
                stats["skipped_comments_not_fetched"] += 1   # don't burn LLM calls on half-context
                continue
            comments = parse_comments(*stored) if stored else []
        context = build_context(p, comments, s, use_comments)
        key = input_hash(system, context, llm.model, s.prompt_version)
        if db.get_extraction(key) is not None:
            stats["cached"] += 1
            continue

        t0 = time.monotonic()
        try:
            res = structured(llm, system, context, Extraction)
        except LLMError as exc:
            db.record_failure(p["id"], "analysis", key, str(exc))
            stats["failed"] += 1
            log.warning("analysis failed for %s: %s", p["id"], exc)
            continue
        latencies.append(time.monotonic() - t0)
        db.save_extraction(key, p["id"], llm.model, s.prompt_version, res.value.model_dump())
        db.resolve_failure(p["id"], "analysis")
        stats["analyzed"] += 1
        stats["needed_retry"] += res.attempts > 1
        ex = res.value
        log.info("[%d] %-14s pain=%-5s int=%d | %s", stats["analyzed"], ex.post_type, ex.has_pain,
                 ex.intensity, p["title"][:60])

    metrics = dict(stats)
    metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 2) if latencies else 0
    metrics["invalid_json_rate"] = round(stats["needed_retry"] / stats["analyzed"], 3) if stats["analyzed"] else 0
    db.finish_run(run_id, metrics)
    return metrics
