"""Pre-fetch layer, step 2: decide which candidate communities are worth fetching.

  rules       free and instant: NSFW/quarantined/private, too small, obvious off-topic ("trading cards")
  profile     one cheap request per survivor: recent post titles + posting rate (kills dead communities)
  classifier  the local LLM judges "is this about trading financial markets?" and picks a category
  review      you confirm in a CSV; nothing is fetched until a community is `approved`
"""
from __future__ import annotations

import csv
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

from .config import Settings
from .fetcher import BlockedError, FetchError, SubredditUnavailable, Transport
from .llm import LLMClient, LLMError, structured
from .models import SubredditClass
from .storage import Database, now

log = logging.getLogger(__name__)

# Reject on off-topic words only when there is NO finance vocabulary at all (the LLM handles the gray zone).
OFFTOPIC_RE = re.compile(
    r"\b(pok[eé]mon|magic the gathering|mtg|yu-?gi-?oh|trading cards?|card trading|tcg|funko|lego|"
    r"steam trad\w*|item trad\w*|skin trad\w*|trade-?ins?|barter\w*|garage sale|buy ?sell ?trade|"
    r"skilled trades?|electrician|plumb\w+|carpent\w+|fantasy (football|basketball|baseball|hockey)|"
    r"trade rumou?rs?)\b", re.I)
FINANCE_RE = re.compile(
    r"\b(stocks?|options?|forex|fx|futures?|crypto\w*|bitcoin|equit\w+|invest\w*|markets?|quant\w*|algo\w*|"
    r"etfs?|securit\w+|brokers?|day ?trad\w+|swing|portfolio|hedge|derivatives?|commodit\w+)\b", re.I)

CLASSIFY_SYSTEM = """You classify Reddit communities for a research project on trading software.
Decide whether the community is mainly about TRADING OR INVESTING IN FINANCIAL MARKETS: stocks, options,
futures, forex, crypto, algorithmic/quant trading, prop-firm trading, general investing, and the tools and
workflows around them. Memes are fine if the community is mainly about market trading.
Say not trading (category "not_trading") for trading cards, game/item trading, barter, trade-ins, skilled
trades, sports trades, or communities that only mention markets in passing.
Pick the single best category. Use only the evidence given; reason is one sentence."""


def rule_reject_reason(row, s: Settings) -> str | None:
    """Instant, no-network checks. Returns a reason string to reject, or None to continue."""
    if row["over18"]:
        return "nsfw"
    if row["quarantine"]:
        return "quarantined"
    if row["subreddit_type"] not in (None, "public", "restricted"):
        return f"type:{row['subreddit_type']}"
    if row["subscribers"] is not None and row["subscribers"] < s.min_subscribers:
        return "too_small"
    head = f"{row['display_name']} {row['title'] or ''} {(row['description'] or '')[:300]}"
    if OFFTOPIC_RE.search(head) and not FINANCE_RE.search(f"{head} {(row['description'] or '')[:1500]}"):
        return "off_topic_keywords"
    return None


def profile_subreddit(transport: Transport, name: str) -> dict:
    """Recent titles + posting rate from one /new request."""
    data = transport.get_json(f"https://www.reddit.com/r/{name}/new.json?limit=25&raw_json=1")
    try:
        posts = [c["data"] for c in data["data"]["children"]]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected listing shape for r/{name}: {exc}") from exc
    stamps = sorted(float(p.get("created_utc", 0)) for p in posts)
    if not stamps:
        return {"n": 0, "titles": [], "posts_per_week": 0.0, "last_activity_utc": 0.0}
    # A full page means the window is [oldest, newest]; a short page means the whole history, up to now.
    window_end = stamps[-1] if len(stamps) >= 25 else time.time()
    days = max((window_end - stamps[0]) / 86400, 1.0)
    return {"n": len(stamps), "titles": [p.get("title", "") for p in posts[:12]],
            "posts_per_week": round(len(stamps) / days * 7, 2), "last_activity_utc": stamps[-1]}


def activity_reject_reason(prof: dict, s: Settings) -> str | None:
    if not prof["n"]:
        return "empty"
    if (time.time() - prof["last_activity_utc"]) / 86400 > s.max_inactive_days:
        return "inactive"
    if prof["posts_per_week"] < s.min_posts_per_week:
        return "low_activity"
    return None


def classify_context(row, prof: dict) -> str:
    titles = "\n".join(f"- {t}" for t in prof["titles"][:10]) or "(none)"
    return (f"NAME: r/{row['display_name']}\nTITLE: {row['title'] or ''}\n"
            f"SUBSCRIBERS: {row['subscribers']}\nPOSTS PER WEEK: {prof['posts_per_week']}\n"
            f"DESCRIPTION:\n{(row['description'] or '')[:700]}\nRECENT POST TITLES:\n{titles}")


def classify_candidates(db: Database, transport: Transport, llm: LLMClient, s: Settings,
                        limit: int = 0, auto_approve: bool = False) -> dict:
    stats: Counter = Counter()
    for row in db.subreddits_to_classify(limit):
        name = row["name"]

        def reject(reason: str, by: str = "rule") -> None:
            db.update_subreddit(name, status="rejected", decided_by=by, reason=reason, classified_at=now())
            stats[f"rejected:{reason}" if by == "rule" else "rejected:llm"] += 1

        reason = rule_reject_reason(row, s)
        if reason:
            reject(reason)
            continue
        try:
            prof = profile_subreddit(transport, name)
        except SubredditUnavailable as exc:
            reject(exc.reason)
            continue
        except BlockedError:
            raise
        except FetchError as exc:
            db.update_subreddit(name, last_error=str(exc)[:200])
            stats["profile_failed"] += 1
            continue
        db.update_subreddit(name, posts_per_week=prof["posts_per_week"], last_activity_utc=prof["last_activity_utc"],
                            sample_titles_json=json.dumps(prof["titles"]))
        reason = activity_reject_reason(prof, s)
        if reason:
            reject(reason)
            continue
        try:
            verdict = structured(llm, CLASSIFY_SYSTEM, classify_context(row, prof), SubredditClass).value
        except LLMError as exc:
            db.update_subreddit(name, last_error=str(exc)[:200])
            stats["llm_failed"] += 1
            continue
        trading = verdict.is_trading_related and verdict.category != "not_trading"
        cols = dict(category=verdict.category, is_trading=int(trading), reason=verdict.reason[:200],
                    classified_at=now(), last_error=None)
        if not trading:
            cols.update(status="rejected", decided_by="llm")
            stats["rejected:llm"] += 1
        elif auto_approve:
            cols.update(status="approved", decided_by="llm")
            stats["approved:llm"] += 1
        else:
            stats["pending_review"] += 1
        db.update_subreddit(name, **cols)
        time.sleep(s.fetch_delay)       # one profile request per community: stay polite
    return dict(stats)


# ---- human review ---------------------------------------------------------------------------
REVIEW_FIELDS = ["name", "subscribers", "posts_per_week", "category", "is_trading", "reason", "status",
                 "discovered_via", "mention_count", "title", "description", "sample_titles", "url", "decision"]


def export_review(db: Database, s: Settings, include_all: bool = False) -> tuple[str, int]:
    """CSV of communities awaiting your decision (or every classified one with include_all)."""
    rows = [r for r in db.subreddits() if (include_all and r["classified_at"]) or
            (r["status"] == "candidate" and r["is_trading"] == 1)]
    path = Path(s.out_dir) / "subreddit_review.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for r in rows:
            titles = json.loads(r["sample_titles_json"] or "[]")
            w.writerow({"name": r["display_name"] or r["name"], "subscribers": r["subscribers"],
                        "posts_per_week": r["posts_per_week"], "category": r["category"],
                        "is_trading": r["is_trading"], "reason": r["reason"], "status": r["status"],
                        "discovered_via": r["discovered_via"], "mention_count": r["mention_count"],
                        "title": r["title"], "description": (r["description"] or "")[:300].replace("\n", " "),
                        "sample_titles": " | ".join(titles[:5]),
                        "url": f"https://www.reddit.com/r/{r['display_name'] or r['name']}", "decision": ""})
    return str(path), len(rows)


APPROVE, REJECT = {"approve", "approved", "yes", "y", "1"}, {"reject", "rejected", "no", "n", "0"}


def apply_review(db: Database, path: str) -> dict:
    """Read the `decision` column (approve / reject; blank = leave as is) and update the registry."""
    stats: Counter = Counter()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            decision = (row.get("decision") or "").strip().lower()
            if not decision:
                continue
            if db.get_subreddit(row["name"]) is None:
                stats["unknown_name"] += 1
            elif decision in APPROVE:
                db.update_subreddit(row["name"], status="approved", decided_by="manual")
                stats["approved"] += 1
            elif decision in REJECT:
                db.update_subreddit(row["name"], status="rejected", decided_by="manual")
                stats["rejected"] += 1
            else:
                stats["unrecognized_decision"] += 1
    return dict(stats)
