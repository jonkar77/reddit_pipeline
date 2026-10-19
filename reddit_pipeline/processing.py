"""Stage 2: processing. Pure functions + one DB-driven entry point. Fully rebuildable from the raw store.

Every post gets a status. Filtered posts are kept (with a reason) so filters are auditable and
tunable, instead of silently disappearing.
"""
from __future__ import annotations

import hashlib
import html
import re
from collections import Counter

from pydantic import ValidationError

from .config import Settings
from .models import RawPost
from .storage import Database

BOT_AUTHORS = {"automoderator", "sneakpeekbot", "remindmebot"}
AD_RE = re.compile(
    r"\b(join (my|our) (discord|telegram|group|community)|free (course|signals?|ebook)|promo code|"
    r"use code|affiliate link|link in bio|dm me for)\b", re.I)
PAIN_HINT_RE = re.compile(
    r"\b(same (problem|issue)|i have the same|annoying|frustrat\w+|wish (there|i|it)|workaround|"
    r"struggl\w+|paying|subscription|too expensive|unreliable)\b", re.I)
URL_RE = re.compile(r"https?://\S+")
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


def normalize_text(s: str | None) -> str:
    s = html.unescape(s or "").replace("\r", "")
    s = ZERO_WIDTH_RE.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def author_key(author: str | None, post_id: str) -> str:
    """Stable pseudonymous key. Deleted authors stay distinct (so they don't collapse into one 'person')."""
    a = (author or "").strip().lower()
    if not a or a == "[deleted]":
        return f"deleted:{post_id}"
    return hashlib.sha256(a.encode()).hexdigest()[:12]


def content_hash(title: str, body: str) -> str:
    text = URL_RE.sub("", f"{title}\n{body}".lower())
    return hashlib.sha1(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def filter_reason(post: RawPost, body: str, s: Settings, sub_status: str | None = None) -> str | None:
    if sub_status in ("candidate", "rejected"):
        return "subreddit_not_approved"     # unknown subs (legacy imports) are treated as approved
    if post.stickied:
        return "stickied"
    if body in ("[removed]", "[deleted]"):
        return "removed"
    if (post.author or "").lower() in BOT_AUTHORS:
        return "bot"
    if AD_RE.search(f"{post.title}\n{body}"):
        return "promo_pattern"
    if post.num_comments < s.min_comments and len(body) < s.min_body_chars:
        return "low_signal"       # little discussion and little text
    return None


def process_posts(db: Database, s: Settings) -> dict:
    """raw_posts -> posts. Idempotent; re-run after changing any filter setting."""
    stats: Counter = Counter()
    rows: list[dict] = []
    status_map = db.subreddit_status_map()
    for _, payload in db.iter_raw_posts(s.subreddit or None):
        try:
            post = RawPost.model_validate(payload)
        except ValidationError:
            stats["invalid_schema"] += 1
            continue
        title, body = normalize_text(post.title), normalize_text(post.selftext)
        sub_name = post.subreddit or s.subreddit or "unknown"
        reason = filter_reason(post, body, s, status_map.get(sub_name.lower()))
        rows.append({
            "id": post.id, "subreddit": sub_name, "title": title, "selftext": body,
            "flair": post.link_flair_text or "", "author_hash": author_key(post.author, post.id),
            "score": post.score, "num_comments": post.num_comments,
            "engagement": post.score + 2 * post.num_comments, "created_utc": post.created_utc,
            "permalink": post.permalink, "content_hash": content_hash(title, body),
            "status": "filtered" if reason else "eligible", "filter_reason": reason,
        })

    # Dedupe reposts/crossposts: keep the highest-engagement copy of identical content.
    best: dict[str, dict] = {}
    for r in rows:
        if r["status"] == "eligible":
            cur = best.get(r["content_hash"])
            if cur is None or r["engagement"] > cur["engagement"]:
                best[r["content_hash"]] = r
    for r in rows:
        if r["status"] == "eligible" and best[r["content_hash"]] is not r:
            r["status"], r["filter_reason"] = "filtered", "duplicate"

    for r in rows:
        stats[r["status"] if r["status"] == "eligible" else f"filtered:{r['filter_reason']}"] += 1
    db.save_posts(rows)
    return dict(stats)


# ---- comments -> LLM context ------------------------------------------------
def _flatten(children: list, out: list) -> None:
    for c in children:
        if c.get("kind") != "t1":
            continue
        d = c["data"]
        out.append({"author": d.get("author") or "", "score": d.get("score", 0), "body": d.get("body", "")})
        replies = d.get("replies")
        if isinstance(replies, dict):
            _flatten(replies["data"]["children"], out)


def parse_comments(payload, fmt: str) -> list[dict]:
    """'flat' = already [{author, score, body}] (imported); 'api' = raw /comments response."""
    if fmt == "flat":
        return list(payload)
    out: list[dict] = []
    try:
        _flatten(payload[1]["data"]["children"], out)
    except (KeyError, IndexError, TypeError):
        pass
    return out


def select_comments(comments: list[dict], k: int, min_len: int = 25) -> list[dict]:
    good = [c for c in comments
            if c["body"] not in ("[deleted]", "[removed]")
            and (c["author"] or "").lower() not in BOT_AUTHORS
            and not (c["author"] or "").lower().endswith("bot")
            and len(c["body"]) >= min_len]
    # relevance: upvotes, nudged up when the comment reads like a problem/workaround/"me too"
    good.sort(key=lambda c: c["score"] + (10 if PAIN_HINT_RE.search(c["body"]) else 0), reverse=True)
    return good[:k]


def build_context(post, comments: list[dict] | None, s: Settings, use_comments: bool = True) -> str:
    """Post + most relevant comments, trimmed to fit the local model's context window."""
    parts = [
        f"SUBREDDIT: r/{post['subreddit']}",
        f"TITLE: {post['title']}",
        f"FLAIR: {post['flair'] or 'none'}",
        f"UPVOTES: {post['score']}  COMMENTS: {post['num_comments']}",
        f"BODY:\n{post['selftext'][:s.max_body_chars] or '(no body, link/title-only post)'}",
    ]
    if not use_comments:
        parts.append("TOP COMMENTS: (not used)")
    elif comments:
        parts.append("TOP COMMENTS:")
        for c in select_comments(comments, s.top_comments):
            parts.append(f"- ({c['score']} pts) {normalize_text(c['body'])[:s.max_comment_chars]}")
    else:
        parts.append("TOP COMMENTS: (none)")
    return "\n".join(parts)[:s.max_context_chars]
