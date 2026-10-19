"""Central configuration. Secrets come from the environment (or a .env file), never from code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader (no dependency). Real environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    # storage
    db_path: str = "data/pipeline.db"
    out_dir: str = "output"
    # reddit
    subreddit: str = ""          # "" = every approved subreddit; a name restricts the run to that one
    reddit_cookie: str = ""
    user_agent: str = "PostmanRuntime/7.43.0"
    fetch_delay: float = 1.0
    fetch_retries: int = 4
    max_pages: int = 10          # per subreddit; Reddit listings stop at ~1000 items anyway
    max_age_days: int = 90       # per subreddit: don't page past posts older than this (0 = no limit)
    max_subreddits: int = 0      # fetch at most N approved subreddits per run (0 = all)
    # discovery / curation
    search_pages: int = 3                # pages of 100 per keyword query
    snowball_min_mentions: int = 2       # r/xyz must be mentioned this many times to become a candidate
    min_subscribers: int = 1000          # rule: ignore tiny communities
    max_inactive_days: int = 60          # rule: ignore communities with no post in this long
    min_posts_per_week: float = 1.0      # rule: ignore near-dead communities
    # triage (cheap title+body pass before comments / full analysis)
    use_triage: bool = True
    triage_prompt_version: str = "triage-v1"
    # processing / filtering
    min_comments: int = 3
    min_body_chars: int = 300
    # LLM context building
    top_comments: int = 12
    max_body_chars: int = 2500
    max_comment_chars: int = 400
    max_context_chars: int = 12000
    # LLM
    backend: str = "ollama"      # "ollama" | "coachwhip" (any OpenAI-compatible server)
    model: str = ""              # "" = qwen3.5:9b for ollama, auto-detect for coachwhip
    coachwhip_url: str = "http://127.0.0.1:8090/v1"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = 8192
    llm_timeout: int = 1800
    prompt_version: str = "extract-v2"   # bump when you change the prompt/schema
    # clustering / scoring
    cluster_method: str = "greedy"       # "greedy" | "hdbscan"
    cluster_threshold: float = 0.72
    min_authors: int = 2                 # ignore "opportunities" backed by a single person
    freq_cap_authors: int = 8            # this many unique authors = max frequency score
    engagement_cap_x: float = 50.0       # summed engagement, in units of each post's subreddit median
    communities_cap: int = 4             # this many distinct communities = max spread score
    weight_frequency: float = 0.30
    weight_engagement: float = 0.15
    weight_intensity: float = 0.25
    weight_wtp: float = 0.15
    weight_communities: float = 0.15

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            db_path=os.environ.get("PIPELINE_DB", cls.db_path),
            out_dir=os.environ.get("PIPELINE_OUT", cls.out_dir),
            subreddit=os.environ.get("REDDIT_SUBREDDIT", cls.subreddit),
            reddit_cookie=os.environ.get("REDDIT_COOKIE", ""),
            user_agent=os.environ.get("REDDIT_UA", cls.user_agent),
            backend=os.environ.get("LLM_BACKEND", cls.backend),
            model=os.environ.get("LLM_MODEL", cls.model),
            coachwhip_url=os.environ.get("COACHWHIP_URL", cls.coachwhip_url),
            embed_model=os.environ.get("EMBED_MODEL", cls.embed_model),
        )
