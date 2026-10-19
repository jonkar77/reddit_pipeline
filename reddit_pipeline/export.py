"""Stage 5: export. The database is the source of truth; CSVs are generated views of one run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import Settings
from .storage import Database

OPP_FIELDS = ["rank", "opportunity", "score", "num_communities", "communities", "unique_authors", "num_posts",
              "total_engagement", "avg_intensity", "score_frequency", "score_engagement", "score_intensity",
              "score_wtp", "score_communities",
              "pain_reason", "target_user", "mvp_idea", "existing_alternatives", "monetization",
              "evidence_links", "build_difficulty_manual", "competition_manual"]
REVIEW_FIELDS = ["rank", "opportunity", "subreddit", "post_id", "permalink", "title", "pain_theme", "pain_summary",
                 "intensity", "willingness_to_pay", "cluster_member_ok"]


def export_run(db: Database, s: Settings, run_id: str | None = None) -> dict:
    run_id = run_id or db.latest_run_id("opportunities")
    if not run_id:
        raise RuntimeError("No opportunities run found. Run the `opportunities` stage first.")
    run = db.get_run(run_id)
    out = Path(s.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opps = db.get_opportunities(run_id)

    ranked = out / "opportunities_ranked.csv"
    with ranked.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OPP_FIELDS)
        w.writeheader()
        for o in opps:
            b = json.loads(o["breakdown_json"])
            w.writerow({**{k: o[k] for k in OPP_FIELDS if k in o},
                        "score_frequency": b["frequency"], "score_engagement": b["engagement"],
                        "score_intensity": b["intensity"], "score_wtp": b["wtp"],
                        "score_communities": b.get("communities", 0),
                        "build_difficulty_manual": "", "competition_manual": ""})

    # Cluster review sheet: hand-check whether members really belong together (merge/split errors).
    records = {r["id"]: r for r in db.analysis_records(run["model"], run["prompt_version"], s.subreddit or None)}
    review = out / "cluster_review.csv"
    with review.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for o in opps:
            for pid in json.loads(o["member_ids_json"]):
                r = records.get(pid)
                if r:
                    w.writerow({"rank": o["rank"], "opportunity": o["opportunity"],
                                "subreddit": r["subreddit"], "post_id": pid,
                                "permalink": "https://www.reddit.com" + r["permalink"], "title": r["title"],
                                "pain_theme": r["pain_theme"], "pain_summary": r["pain_summary"],
                                "intensity": r["intensity"], "willingness_to_pay": r["willingness_to_pay"],
                                "cluster_member_ok": ""})
    return {"run_id": run_id, "opportunities": len(opps), "ranked_csv": str(ranked), "review_csv": str(review)}
