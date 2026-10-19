"""Data contracts. Everything crossing a stage boundary is validated against one of these."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

PostType = Literal[
    "pain_point",      # concrete problem / frustration
    "tool_request",    # asks for a tool/feature that doesn't exist
    "question_help",   # how-to / debugging / beginner question
    "discussion",      # opinions, no concrete unmet need
    "showcase_or_ad",  # self-promotion, product plug, course, signal service
    "results_flex",    # P&L screenshots, "I made $X"
    "news_or_meme",
    "other",
]

# Evidence strength for willingness to pay (scored in scoring.py, not by the LLM)
WtpSignal = Literal[
    "none",                 # no mention of paying
    "would_pay",            # someone says they'd pay / hire for this
    "paid_tool_mentioned",  # names a paid tool/service they use or tried for this
    "explicit_price",       # states a concrete price/cost they pay or would pay
]


class RawPost(BaseModel):
    """Boundary check for one Reddit listing child. Extra fields are ignored."""
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str = ""
    selftext: str = ""
    author: str | None = None
    link_flair_text: str | None = None
    subreddit: str = ""
    score: int = 0
    num_comments: int = 0
    created_utc: float = 0.0
    permalink: str = ""
    stickied: bool = False


class Triage(BaseModel):
    """Cheap first pass over title+body only: is this post worth comments + full analysis?"""
    worth_analyzing: bool        # could plausibly contain a concrete tooling/workflow/cost problem
    post_type: PostType
    reason: str                  # one short phrase


SubCategory = Literal[
    "equities", "options", "futures", "forex", "crypto", "algo_quant", "prop_firms",
    "general_investing", "education_beginners", "other_trading", "not_trading",
]


class RawSubreddit(BaseModel):
    """Boundary check for a subreddit listing/about payload. Reddit often returns nulls here."""
    model_config = ConfigDict(extra="ignore")

    display_name: str
    title: str | None = None
    public_description: str | None = None
    description: str | None = None
    subscribers: int | None = None
    created_utc: float | None = None
    over18: bool | None = None
    subreddit_type: str | None = None
    quarantine: bool | None = None


class SubredditClass(BaseModel):
    """Local-LLM verdict on whether a community is about trading financial markets."""
    is_trading_related: bool
    category: SubCategory
    reason: str                  # one sentence


class Extraction(BaseModel):
    """What the local LLM must produce for each post/thread."""
    post_type: PostType
    has_pain: bool
    pain_summary: str            # 1-2 sentences, the concrete problem. "" if none
    pain_theme: str              # 3-8 word noun phrase, e.g. "unreliable free historical data"
    who_feels_it: str            # e.g. "retail quant with Python backtests"
    intensity: int               # 1 (mild annoyance) .. 5 (blocking / costly / repeated)
    workaround: str              # what people do today. "" if none
    software_opportunity: str    # one sentence. "" if none
    buildable_as_software: bool
    willingness_to_pay: WtpSignal
    evidence: list[str]          # up to 2 short verbatim quotes

    @field_validator("intensity", mode="before")
    @classmethod
    def _clamp_intensity(cls, v):
        try:
            return max(1, min(5, int(v)))
        except (TypeError, ValueError):
            return v

    @field_validator("evidence")
    @classmethod
    def _trim_evidence(cls, v: list[str]) -> list[str]:
        return [e[:300] for e in v[:2]]


class Opportunity(BaseModel):
    """What the LLM writes for one cluster of similar pains."""
    opportunity: str             # "Tool that <does what> for <whom>"
    pain_reason: str             # 2-3 sentences
    target_user: str
    mvp_idea: str                # smallest useful first version
    existing_alternatives: str   # tools people mention, or "none mentioned"
    monetization: str
