"""Command-line entry point: one subcommand per pipeline stage, plus `run-all`.

  discovery (pre-fetch layer)
    python -m reddit_pipeline discover --seed algotrading,options
    python -m reddit_pipeline classify
    python -m reddit_pipeline review-export     # edit the `decision` column, then:
    python -m reddit_pipeline review-apply output/subreddit_review.csv

  pipeline
    python -m reddit_pipeline fetch             # every approved subreddit
    python -m reddit_pipeline process
    python -m reddit_pipeline triage
    python -m reddit_pipeline comments --limit 200
    python -m reddit_pipeline analyze --limit 5          # quick trial
    python -m reddit_pipeline opportunities
    python -m reddit_pipeline export
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from dataclasses import replace

from .config import Settings
from .storage import Database

# CLI arg name -> Settings field
OVERRIDES = {
    "db": "db_path", "out_dir": "out_dir", "subreddit": "subreddit", "backend": "backend", "model": "model",
    "base_url": "coachwhip_url", "cluster_method": "cluster_method", "threshold": "cluster_threshold",
    "min_authors": "min_authors", "max_pages": "max_pages", "top_comments": "top_comments",
    "max_subs": "max_subreddits", "max_age_days": "max_age_days", "min_subscribers": "min_subscribers",
    "pages": "search_pages",
}


def build_settings(args: argparse.Namespace) -> Settings:
    overrides = {field: getattr(args, arg) for arg, field in OVERRIDES.items()
                 if getattr(args, arg, None) is not None}
    if getattr(args, "no_triage", False):
        overrides["use_triage"] = False
    return replace(Settings.from_env(), **overrides)


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="SQLite file (default data/pipeline.db)")
    common.add_argument("--out-dir", help="where exports go (default output/)")
    common.add_argument("--subreddit", help="restrict the run to one subreddit (default: all approved)")
    common.add_argument("--no-triage", action="store_true", help="skip the triage gate (comments/analyze use every eligible post)")
    common.add_argument("--backend", choices=["ollama", "coachwhip"], help="LLM backend (default ollama)")
    common.add_argument("--model", help="model name (default qwen3.5:9b; coachwhip auto-detects)")
    common.add_argument("--base-url", help="OpenAI-compatible server URL for the coachwhip backend")
    common.add_argument("-v", "--verbose", action="store_true")

    cluster = argparse.ArgumentParser(add_help=False)
    cluster.add_argument("--cluster-method", choices=["greedy", "hdbscan"])
    cluster.add_argument("--threshold", type=float, help="cosine similarity to merge pains (greedy)")
    cluster.add_argument("--min-authors", type=int, help="drop opportunities with fewer unique authors")

    p = argparse.ArgumentParser(prog="reddit_pipeline", description="Reddit pain-point research pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("import-json", parents=[common], help="import existing response_*.json / comment caches")
    c.add_argument("--posts-dir", default="reddit-test-data")
    c.add_argument("--comments-dir", help="old analysis/comments/ cache of flat comment lists")

    c = sub.add_parser("discover", parents=[common], help="pre-fetch: find candidate communities (search, snowball, seeds)")
    c.add_argument("--queries", help="comma-separated keywords (default: built-in trading list)")
    c.add_argument("--seed", help="comma-separated subreddits to approve immediately")
    c.add_argument("--no-snowball", action="store_true", help="skip mining r/xyz mentions and sidebars")
    c.add_argument("--pages", type=int, help="pages of 100 per keyword query (default 3)")
    c.add_argument("--hydrate-limit", type=int, default=50, help="max metadata lookups for snowballed names")

    c = sub.add_parser("classify", parents=[common], help="pre-fetch: rules + activity + local-LLM relevance check")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--auto-approve", action="store_true", help="approve LLM-confirmed trading communities without review")
    c.add_argument("--min-subscribers", type=int)

    c = sub.add_parser("review-export", parents=[common], help="write subreddit_review.csv for your decisions")
    c.add_argument("--all", action="store_true", help="include already-decided communities too")

    c = sub.add_parser("review-apply", parents=[common], help="apply the `decision` column from the review CSV")
    c.add_argument("file")

    c = sub.add_parser("subreddits", parents=[common], help="list the registry, or approve/reject by name")
    c.add_argument("--status", choices=["candidate", "approved", "rejected"])
    c.add_argument("--approve", help="comma-separated names")
    c.add_argument("--reject", help="comma-separated names")

    c = sub.add_parser("fetch", parents=[common], help="stage 1: fetch new posts from every approved subreddit")
    c.add_argument("--backfill", action="store_true", help="walk each listing as deep as allowed, resuming from checkpoint")
    c.add_argument("--max-pages", type=int, help="pages per subreddit")
    c.add_argument("--max-subs", type=int, help="fetch at most N subreddits this run")
    c.add_argument("--max-age-days", type=int, help="ignore posts older than this (0 = no limit)")

    sub.add_parser("process", parents=[common], help="stage 2: validate, normalize, dedupe, filter")

    c = sub.add_parser("triage", parents=[common], help="stage 2b: cheap title+body keep/drop pass (gates comments + analysis)")
    c.add_argument("--limit", type=int, default=0, help="max NEW LLM calls this run")

    c = sub.add_parser("comments", parents=[common], help="fetch comment threads for eligible posts")
    c.add_argument("--limit", type=int, default=0)

    c = sub.add_parser("analyze", parents=[common], help="stage 3: local LLM extraction (cached, resumable)")
    c.add_argument("--limit", type=int, default=0, help="max NEW LLM calls this run")
    c.add_argument("--skip-comments", action="store_true", help="analyse post text only")
    c.add_argument("--top-comments", type=int)

    sub.add_parser("retry-failed", parents=[common], help="replay the failed-record queue")
    sub.add_parser("opportunities", parents=[common, cluster], help="stage 4: cluster, synthesize, score")

    c = sub.add_parser("export", parents=[common], help="stage 5: write CSVs for a run")
    c.add_argument("--run-id")

    c = sub.add_parser("eval-sample", parents=[common], help="write a random sample for hand-labelling")
    c.add_argument("--n", type=int, default=50)
    c.add_argument("--path", default="output/eval_sample.csv")

    c = sub.add_parser("eval-score", parents=[common], help="score extractions against your labels")
    c.add_argument("labels")

    sub.add_parser("stats", parents=[common], help="row counts, filter reasons, failures")

    c = sub.add_parser("run-all", parents=[common, cluster],
                       help="process -> triage -> comments -> analyze -> opportunities -> export")
    c.add_argument("--limit", type=int, default=0, help="max NEW LLM calls for the analyze stage")
    c.add_argument("--fetch", action="store_true", help="also run the fetch stage first")
    return p


def import_json(db: Database, s: Settings, posts_dir: str, comments_dir: str | None) -> dict:
    posts = comments = 0
    subs: set[str] = set()
    for path in sorted(glob.glob(os.path.join(posts_dir, "response_*.json"))):
        with open(path, encoding="utf-8") as fh:
            for child in json.load(fh)["data"]["children"]:
                d = child["data"]
                name = d.get("subreddit") or s.subreddit or "unknown"
                db.upsert_raw_post(d["id"], name, d)
                subs.add(name)
                posts += 1
    for name in subs:                      # imported communities are trusted: register them as approved
        db.upsert_subreddit(name, via="import", status="approved", decided_by="import")
    if comments_dir:
        for path in glob.glob(os.path.join(comments_dir, "*.json")):
            with open(path, encoding="utf-8") as fh:
                db.save_comments(os.path.splitext(os.path.basename(path))[0], json.load(fh), "flat")
            comments += 1
    return {"posts": posts, "comment_threads": comments}


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    s = build_settings(args)
    db = Database(s.db_path)
    try:
        run(args, s, db)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run the same command to resume.")
        sys.exit(130)
    finally:
        db.close()


def make_transport(s: Settings):
    from .fetcher import CurlTransport
    return CurlTransport(s.user_agent, s.reddit_cookie, s.fetch_retries)


def run(args: argparse.Namespace, s: Settings, db: Database) -> None:
    # Heavy imports are lazy so `stats`, `process`, etc. work without LLM dependencies.
    cmd = args.cmd
    names = lambda v: [x.strip().removeprefix("r/") for x in (v or "").split(",") if x.strip()]  # noqa: E731

    if cmd == "import-json":
        print(import_json(db, s, args.posts_dir, args.comments_dir))
    elif cmd == "discover":
        from .discovery import run_discovery
        print(run_discovery(db, make_transport(s), s, queries=names(args.queries) or None, seeds=names(args.seed),
                            do_snowball=not args.no_snowball, hydrate_limit=args.hydrate_limit))
    elif cmd == "classify":
        from .curation import classify_candidates
        from .llm import make_llm
        print(classify_candidates(db, make_transport(s), make_llm(s), s, args.limit, args.auto_approve))
    elif cmd == "review-export":
        from .curation import export_review
        path, n = export_review(db, s, args.all)
        print(f"wrote {n} communities to {path}. Fill the `decision` column (approve/reject), then run review-apply.")
    elif cmd == "review-apply":
        from .curation import apply_review
        print(apply_review(db, args.file), "(run `process` so posts from newly approved/rejected communities update)")
    elif cmd == "subreddits":
        for name in names(args.approve):
            db.upsert_subreddit(name, via="manual", status="approved", decided_by="manual", force_status=True)
        for name in names(args.reject):
            db.upsert_subreddit(name, via="manual", status="rejected", decided_by="manual", force_status=True)
        for r in db.subreddits(args.status):
            print(f"{r['status']:<9} {r['name']:<28} subs={r['subscribers']} cat={r['category']} via={r['discovered_via']}")
    elif cmd == "fetch" or (cmd == "run-all" and args.fetch):
        from .fetcher import fetch_all
        print(fetch_all(db, make_transport(s), s, backfill=getattr(args, "backfill", False)))

    if cmd in ("process", "run-all"):
        from .processing import process_posts
        print(process_posts(db, s))
    if cmd in ("triage", "run-all") and s.use_triage:
        from .llm import make_llm
        from .triage import run_triage
        print(run_triage(db, make_llm(s), s, getattr(args, "limit", 0)))
    if cmd in ("comments", "run-all"):
        from .fetcher import fetch_comments
        print(fetch_comments(db, make_transport(s), s, getattr(args, "limit", 0) if cmd == "comments" else 0))
    if cmd in ("analyze", "retry-failed", "run-all"):
        from .analysis import run_analysis
        from .llm import make_llm
        print(run_analysis(db, make_llm(s), s, limit=getattr(args, "limit", 0),
                           use_comments=not getattr(args, "skip_comments", False),
                           only_failed=(cmd == "retry-failed")))
    if cmd in ("opportunities", "run-all"):
        from .llm import make_embedder, make_llm
        from .opportunities import build_opportunities
        run_id, stats = build_opportunities(db, make_llm(s), make_embedder(s), s)
        print(run_id, stats)
    if cmd in ("export", "run-all"):
        from .export import export_run
        print(export_run(db, s, getattr(args, "run_id", None)))
    if cmd == "eval-sample":
        from .evaluate import make_sample
        print(f"wrote {make_sample(db, s, args.n, args.path)} rows to {args.path}")
    if cmd == "eval-score":
        from .evaluate import score_labels
        print(score_labels(db, s, args.labels))
    if cmd == "stats":
        from collections import Counter
        failed = Counter(f["stage"] for f in db.unresolved_failures())
        print(json.dumps({**db.counts(), "filter_reasons": db.filter_reasons(),
                          "failed_by_stage": dict(failed)}, indent=2))
