"""Rule-based, reproducible opportunity score. No LLM in the loop, every component is inspectable.

Design choices (vs. asking the model for a score):
  frequency   counts UNIQUE AUTHORS, not posts, so one loud person can't dominate
  engagement  is measured in units of each post's OWN subreddit median, so big communities don't win by size
  intensity   comes from the extracted 1-5 field
  wtp         is graded by evidence strength (explicit price > paid tool > "would pay" > none)
  communities how many DISTINCT communities share the pain (1 community = 0, `communities_cap` = max)
Build difficulty / competition are deliberately NOT scored: annotate those by hand for your top picks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Settings

WTP_STRENGTH = {"none": 0.0, "would_pay": 0.5, "paid_tool_mentioned": 0.7, "explicit_price": 1.0}


@dataclass
class Score:
    total: float            # 0..10
    components: dict        # each 0..1, plus raw counts


def score_cluster(members: list[dict], medians: dict[str, float], s: Settings) -> Score:
    authors = {m["author_hash"] for m in members}
    communities = sorted({(m.get("subreddit") or "").lower() for m in members} - {""})
    total_eng = sum(m["engagement"] for m in members)

    frequency = min(1.0, math.log1p(len(authors)) / math.log1p(s.freq_cap_authors))
    relative = sum(m["engagement"] / max(medians.get((m.get("subreddit") or "").lower(), 1.0), 1.0)
                   for m in members)
    engagement = min(1.0, math.log1p(relative) / math.log1p(s.engagement_cap_x))
    intensity = (sum(m["intensity"] for m in members) / len(members)) / 5
    wtp = sum(WTP_STRENGTH.get(m.get("willingness_to_pay", "none"), 0.0) for m in members) / len(members)
    spread = min(1.0, max(0, len(communities) - 1) / max(s.communities_cap - 1, 1))

    total = 10 * (s.weight_frequency * frequency + s.weight_engagement * engagement
                  + s.weight_intensity * intensity + s.weight_wtp * wtp + s.weight_communities * spread)
    return Score(round(total, 2), {
        "frequency": round(frequency, 3), "engagement": round(engagement, 3),
        "intensity": round(intensity, 3), "wtp": round(wtp, 3), "communities": round(spread, 3),
        "unique_authors": len(authors), "total_engagement": total_eng,
        "num_communities": len(communities), "community_names": communities,
    })
