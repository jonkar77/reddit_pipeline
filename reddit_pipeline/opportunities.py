"""Stage 4: cluster per-post pains into opportunities, synthesize each, and score it."""
from __future__ import annotations

import json
import logging
from collections import Counter

from .clustering import cluster_records
from .config import Settings
from .llm import Embedder, LLMClient, LLMError, structured
from .models import Opportunity
from .scoring import score_cluster
from .storage import Database

log = logging.getLogger(__name__)

SYNTH_SYSTEM = """You are a product strategist. You receive several posts from trading-related Reddit
communities that all describe the same underlying pain. Define ONE software opportunity that addresses it.
Be specific and grounded ONLY in the provided evidence. `opportunity` should read like
"Tool that <does what> for <whom>"."""


def is_actionable(r: dict) -> bool:
    return bool(r["has_pain"] and r["buildable_as_software"] and r["intensity"] >= 2
                and r["software_opportunity"].strip())


def synthesize(llm: LLMClient, group: list[dict], s: Settings) -> Opportunity:
    top = sorted(group, key=lambda r: r["engagement"], reverse=True)[:10]
    lines = [f"- Community: r/{g['subreddit']}\n  Pain: {g['pain_summary']}\n  Who: {g['who_feels_it']}\n"
             f"  Workaround: {g['workaround'] or 'none'}\n  Idea: {g['software_opportunity']}" for g in top]
    return structured(llm, SYNTH_SYSTEM, "\n".join(lines), Opportunity).value


def build_opportunities(db: Database, llm: LLMClient, embedder: Embedder | None, s: Settings) -> tuple[str, dict]:
    records = db.analysis_records(llm.model, s.prompt_version, s.subreddit or None)
    pains = [r for r in records if is_actionable(r)]
    run_id = db.start_run("opportunities", llm.model, s.prompt_version, {
        "cluster_method": s.cluster_method, "threshold": s.cluster_threshold, "min_authors": s.min_authors})
    stats: Counter = Counter(records=len(records), actionable=len(pains))

    rows: list[dict] = []
    if pains:
        pains.sort(key=lambda r: r["engagement"], reverse=True)
        groups = cluster_records(pains, embedder, s.cluster_method, s.cluster_threshold)
        stats["clusters"] = len(groups)
        medians = db.median_engagements()
        for g in groups:
            if len({m["author_hash"] for m in g}) < s.min_authors:
                stats["dropped_min_authors"] += 1
                continue
            try:
                opp = synthesize(llm, g, s)
            except LLMError as exc:
                stats["synthesis_failed"] += 1
                log.warning("synthesis failed: %s", exc)
                continue
            sc = score_cluster(g, medians, s)
            top3 = sorted(g, key=lambda r: r["engagement"], reverse=True)[:3]
            rows.append({
                **opp.model_dump(),
                "score": sc.total, "num_posts": len(g),
                "unique_authors": sc.components["unique_authors"],
                "total_engagement": sc.components["total_engagement"],
                "avg_intensity": round(sum(m["intensity"] for m in g) / len(g), 2),
                "breakdown_json": json.dumps(sc.components),
                "member_ids_json": json.dumps([m["id"] for m in g]),
                "evidence_links": " ".join("https://www.reddit.com" + m["permalink"] for m in top3),
                "num_communities": sc.components["num_communities"],
                "communities": ", ".join("r/" + c for c in sc.components["community_names"]),
            })
        rows.sort(key=lambda r: (r["score"], r["total_engagement"]), reverse=True)
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        db.save_opportunities(run_id, rows)
    stats["opportunities"] = len(rows)
    db.finish_run(run_id, dict(stats))
    return run_id, dict(stats)
