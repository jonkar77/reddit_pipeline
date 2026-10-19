"""Offline end-to-end tests: stub LLM, stub embedder, fake Reddit. Run: python -m unittest -v"""
import csv
import json
import re
import tempfile
import time
import unittest
import zlib
from dataclasses import replace
from urllib.parse import parse_qs, urlparse

import numpy as np

from reddit_pipeline.analysis import run_analysis
from reddit_pipeline.config import Settings
from reddit_pipeline.curation import apply_review, classify_candidates, export_review
from reddit_pipeline.discovery import extract_mentions, run_discovery
from reddit_pipeline.export import export_run
from reddit_pipeline.fetcher import SubredditUnavailable, fetch_all, fetch_posts
from reddit_pipeline.opportunities import build_opportunities
from reddit_pipeline.processing import process_posts
from reddit_pipeline.scoring import score_cluster
from reddit_pipeline.storage import Database
from reddit_pipeline.triage import run_triage

LONG = "I keep running into this and it costs me hours every week. " * 8
NOW = time.time()


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def raw(pid, title, body=LONG, score=10, comments=6, author="u", created=1000.0, **extra):
    return {"id": pid, "title": title, "selftext": body, "author": author, "subreddit": "algotrading",
            "score": score, "num_comments": comments, "created_utc": created,
            "permalink": f"/r/algotrading/comments/{pid}/x/", "stickied": False, **extra}


def sub_data(name, subs, desc="", title="", **extra):
    return {"display_name": name, "title": title or name, "public_description": desc, "description": "",
            "subscribers": subs, "created_utc": 1.0, "over18": False, "subreddit_type": "public",
            "quarantine": False, **extra}


class StubLLM:
    model = "stub"

    def __init__(self, garbage_first=False, always_garbage=False):
        self.calls, self.garbage_first, self.always_garbage = 0, garbage_first, always_garbage

    def complete(self, system, user, schema, temperature):
        self.calls += 1
        if self.always_garbage or (self.garbage_first and self.calls == 1):
            return "sorry, no JSON here"
        low, kind = user.lower(), schema.__name__
        if kind == "Opportunity":
            return json.dumps({"opportunity": "Tool that fixes a thing for quants", "pain_reason": "Hard.",
                               "target_user": "retail quant", "mvp_idea": "CLI",
                               "existing_alternatives": "none mentioned", "monetization": "subscription"})
        if kind == "Triage":
            keep = "data" in low or "backtest" in low
            return json.dumps({"worth_analyzing": keep, "post_type": "pain_point" if keep else "discussion",
                               "reason": "stub"})
        if kind == "SubredditClass":
            trading = any(w in low for w in ("forex", "options", "stocks"))
            return json.dumps({"is_trading_related": trading, "category": "options" if trading else "not_trading",
                               "reason": "stub"})
        theme = "expensive historical data" if "data" in low else ("slow backtesting" if "backtest" in low else "")
        return json.dumps({
            "post_type": "pain_point" if theme else "discussion", "has_pain": bool(theme),
            "pain_summary": f"people struggle with {theme}" if theme else "", "pain_theme": theme,
            "who_feels_it": "retail quant", "intensity": 4 if theme else 1, "workaround": "scripts",
            "software_opportunity": "a tool" if theme else "", "buildable_as_software": bool(theme),
            "willingness_to_pay": "explicit_price" if "data" in low else "none", "evidence": ["quote"]})


class StubEmbedder:
    """Hashed bag-of-words: identical themes -> identical vectors, unrelated themes -> ~orthogonal."""

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


class PagedTransport:
    """Serves a fixed list of listing pages, for pagination/incremental tests."""

    def __init__(self, pages):
        self.pages = pages

    def get_json(self, url):
        idx = 0 if "after=" not in url else int(url.split("after=")[1])
        children = [{"kind": "t3", "data": d} for d in self.pages[idx]]
        return {"data": {"children": children, "after": str(idx + 1) if idx + 1 < len(self.pages) else None}}


class FakeReddit:
    """Routes the URLs the pipeline uses: subreddit search, about.json, and /new.json (posts + profiling)."""

    def __init__(self):
        self.search, self.about, self.listing = {}, {}, {}

    @staticmethod
    def _resolve(value):
        if isinstance(value, Exception):
            raise value
        return value

    def get_json(self, url):
        u = urlparse(url)
        q = parse_qs(u.query)
        if u.path == "/subreddits/search.json":
            items = self.search.get(q["q"][0], [])
            return {"data": {"children": [{"kind": "t5", "data": d} for d in items], "after": None}}
        m = re.match(r"/r/([^/]+)/about\.json", u.path)
        if m:
            return {"data": self._resolve(self.about[m.group(1).lower()])}
        m = re.match(r"/r/([^/]+)/new\.json", u.path)
        if m:
            posts = self._resolve(self.listing.get(m.group(1).lower(), []))[: int(q["limit"][0])]
            return {"data": {"children": [{"kind": "t3", "data": d} for d in posts], "after": None}}
        raise AssertionError(f"unexpected URL {url}")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Settings(db_path=":memory:", out_dir=self.tmp.name, fetch_delay=0, model="stub",
                          max_age_days=0, use_triage=False)
        self.db = Database(":memory:")

    def seed(self):
        posts = [
            raw("a1", "Historical data is too expensive", author="u1", score=50, comments=30),
            raw("a2", "Where do I get cheap data?", author="u2", score=20, comments=10),
            raw("b1", "Backtests take forever", author="u3", score=30, comments=12),
            raw("b2", "Backtests are slow again", author="u3", score=5, comments=4),   # same author as b1
            raw("ad", "Join my discord for free signals", author="spam"),
            raw("dup", "Historical data is too expensive", author="u9", score=1, comments=3),  # repost of a1
            raw("low", "gm", body="short", comments=0),
            raw("rm", "removed", body="[removed]"),
        ]
        for p in posts:
            self.db.upsert_raw_post(p["id"], "algotrading", p)
        for pid in ("a1", "a2", "b1", "b2", "ad", "dup"):
            self.db.save_comments(pid, [{"author": "x", "score": 9, "body": "Same problem here, it is annoying"}], "flat")

    # ---- stage 2: processing ----
    def test_processing_filters_and_dedupes(self):
        self.seed()
        stats = process_posts(self.db, self.s)
        self.assertEqual(stats["eligible"], 4)
        self.assertEqual(stats["filtered:promo_pattern"], 1)
        self.assertEqual(stats["filtered:duplicate"], 1)
        self.assertEqual(stats["filtered:low_signal"], 1)
        self.assertEqual(stats["filtered:removed"], 1)
        self.assertEqual(process_posts(self.db, self.s), stats)          # idempotent

    def test_processing_respects_subreddit_approval(self):
        self.db.upsert_subreddit("rejectedsub", via="search:x", status="rejected", decided_by="manual")
        self.db.upsert_subreddit("pendingsub", via="search:x")           # candidate: not yet approved
        for pid, sub in (("r1", "rejectedsub"), ("p1", "pendingsub"), ("l1", "legacysub")):
            self.db.upsert_raw_post(pid, sub, raw(pid, f"Data pain {pid}", subreddit=sub))
        stats = process_posts(self.db, self.s)
        self.assertEqual(stats["filtered:subreddit_not_approved"], 2)
        self.assertEqual(stats["eligible"], 1)                           # unknown (legacy) subs stay eligible

    # ---- stage 3: analysis ----
    def test_analysis_cache_retry_and_failure_queue(self):
        self.seed()
        process_posts(self.db, self.s)
        llm = StubLLM(garbage_first=True)
        m1 = run_analysis(self.db, llm, self.s)
        self.assertEqual((m1["analyzed"], m1["needed_retry"]), (4, 1))   # 1 bad reply retried and recovered
        calls = llm.calls
        m2 = run_analysis(self.db, llm, self.s)
        self.assertEqual((m2.get("analyzed", 0), m2["cached"]), (0, 4))  # second run is free
        self.assertEqual(llm.calls, calls)

    def test_failed_records_are_queued_and_replayable(self):
        self.seed()
        process_posts(self.db, self.s)
        m = run_analysis(self.db, StubLLM(always_garbage=True), self.s)
        self.assertEqual(m["failed"], 4)
        self.assertEqual(len(self.db.unresolved_failures("analysis")), 4)
        run_analysis(self.db, StubLLM(), self.s, only_failed=True)
        self.assertEqual(len(self.db.unresolved_failures("analysis")), 0)

    def test_posts_without_fetched_comments_are_skipped(self):
        self.db.upsert_raw_post("n1", "algotrading", raw("n1", "Historical data pain", comments=9))
        process_posts(self.db, self.s)
        m = run_analysis(self.db, StubLLM(), self.s)
        self.assertEqual(m["skipped_comments_not_fetched"], 1)

    # ---- triage gate ----
    def test_triage_gates_comments_and_analysis(self):
        s = replace(self.s, use_triage=True)
        for pid, title in (("k1", "Data feeds break constantly"), ("n1", "Anyone watching the game tonight?")):
            self.db.upsert_raw_post(pid, "algotrading", raw(pid, title, comments=9))
        process_posts(self.db, s)
        llm = StubLLM()

        self.assertEqual(run_analysis(self.db, llm, s)["skipped_not_triaged"], 2)   # nothing triaged yet
        t = run_triage(self.db, llm, s)
        self.assertEqual((t["kept"], t["rejected"]), (1, 1))
        self.assertEqual(run_triage(self.db, llm, s)["cached"], 2)                   # triage is cached too

        wanted = [r["id"] for r in self.db.posts_needing_comments("", 0, True, s.triage_prompt_version)]
        self.assertEqual(wanted, ["k1"])                                             # no comment fetch for noise
        self.assertEqual(len(self.db.posts_needing_comments("", 0, False, "")), 2)

        m = run_analysis(self.db, llm, s)
        self.assertEqual(m["skipped_triage_rejected"], 1)
        self.assertEqual(m["skipped_comments_not_fetched"], 1)                       # k1 still needs its comments

    # ---- stage 4/5: opportunities & scoring ----
    def test_opportunities_scoring_and_export(self):
        self.seed()
        process_posts(self.db, self.s)
        llm = StubLLM()
        run_analysis(self.db, llm, self.s)
        run_id, stats = build_opportunities(self.db, llm, StubEmbedder(), self.s)
        # data cluster has 2 unique authors; backtest cluster has 1 (b1+b2 same author) -> dropped
        self.assertEqual(stats["opportunities"], 1)
        self.assertEqual(stats["dropped_min_authors"], 1)
        info = export_run(self.db, self.s, run_id)
        rows = read_csv(info["ranked_csv"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["unique_authors"], "2")
        self.assertEqual(rows[0]["num_communities"], "1")
        self.assertIn("build_difficulty_manual", rows[0])
        review = read_csv(info["review_csv"])
        self.assertEqual(len(review), 2)

    def test_same_pain_across_communities_is_ranked_and_labelled(self):
        for pid, sub, author in (("x1", "algotrading", "u1"), ("x2", "quant", "u2"), ("x3", "options", "u3")):
            self.db.upsert_subreddit(sub, via="seed", status="approved", decided_by="seed")
            self.db.upsert_raw_post(pid, sub, raw(pid, f"Historical data costs too much ({sub})", author=author, subreddit=sub))
            self.db.save_comments(pid, [], "flat")
        process_posts(self.db, self.s)
        llm = StubLLM()
        run_analysis(self.db, llm, self.s)
        run_id, stats = build_opportunities(self.db, llm, StubEmbedder(), self.s)
        self.assertEqual(stats["opportunities"], 1)
        row = read_csv(export_run(self.db, self.s, run_id)["ranked_csv"])[0]
        self.assertEqual(row["num_communities"], "3")
        self.assertEqual(sorted(row["communities"].split(", ")), ["r/algotrading", "r/options", "r/quant"])

    def test_scoring_rewards_unique_authors_payment_and_spread(self):
        base = {"engagement": 40, "intensity": 3, "willingness_to_pay": "none", "subreddit": "a"}
        med = {"a": 10.0, "b": 10.0}
        score = lambda m: score_cluster(m, med, self.s).total  # noqa: E731
        one_author = [{**base, "author_hash": "x"}, {**base, "author_hash": "x"}]
        two_authors = [{**base, "author_hash": "x"}, {**base, "author_hash": "y"}]
        paid = [{**m, "willingness_to_pay": "explicit_price"} for m in two_authors]
        spread = [two_authors[0], {**two_authors[1], "subreddit": "b"}]
        self.assertLess(score(one_author), score(two_authors))
        self.assertLess(score(two_authors), score(paid))
        self.assertLess(score(two_authors), score(spread))
        self.assertEqual(score_cluster(two_authors, med, self.s).components["communities"], 0.0)
        # the same raw engagement counts for more in a quiet community than in a huge one
        quiet = score_cluster(two_authors, {"a": 5.0}, self.s).components["engagement"]
        busy = score_cluster(two_authors, {"a": 500.0}, self.s).components["engagement"]
        self.assertGreater(quiet, busy)

    # ---- stage 1: fetching ----
    def test_fetch_is_incremental_and_idempotent(self):
        pages = [[raw("p1", "t1", created=300.0), raw("p2", "t2", created=200.0)], [raw("p3", "t3", created=100.0)]]
        first = fetch_posts(self.db, PagedTransport(pages), self.s, "algotrading")
        self.assertEqual((first["pages"], first["new"]), (2, 3))
        second = fetch_posts(self.db, PagedTransport(pages), self.s, "algotrading")   # nothing newer: stops at page 1
        self.assertEqual((second["pages"], second["new"], second["updated"]), (1, 0, 2))
        self.assertEqual(self.db.counts()["raw_posts"], 3)                           # no duplicates

    def test_fetch_validates_schema_at_the_boundary(self):
        stats = fetch_posts(self.db, PagedTransport([[raw("ok", "fine"), {"title": "no id"}]]), self.s, "algotrading")
        self.assertEqual((stats["new"], stats["invalid"]), (1, 1))

    def test_fetch_stops_at_the_age_window(self):
        pages = [[raw("new", "t", created=NOW - 86400), raw("old", "t", created=NOW - 200 * 86400)]]
        stats = fetch_posts(self.db, PagedTransport(pages), replace(self.s, max_age_days=90), "algotrading")
        self.assertEqual(stats["new"], 1)

    def test_fetch_all_covers_approved_subs_and_isolates_failures(self):
        fake = FakeReddit()
        for name in ("ok1", "priv", "ok2", "notapproved"):
            self.db.upsert_subreddit(name, via="seed", status="approved" if name != "notapproved" else None,
                                     decided_by="seed")
        fake.listing["ok1"] = [raw("o1", "t", created=NOW - 100, subreddit="ok1")]
        fake.listing["ok2"] = [raw("o2", "t", created=NOW - 100, subreddit="ok2")]
        fake.listing["priv"] = SubredditUnavailable("private")
        fake.listing["notapproved"] = [raw("n1", "t", created=NOW - 100, subreddit="notapproved")]
        result = fetch_all(self.db, fake, self.s)
        self.assertEqual((result["subreddits"], result["new_posts"]), (3, 2))        # unapproved never fetched
        self.assertEqual(result["errors"], {"priv": "private"})
        self.assertEqual(self.db.get_subreddit("priv")["last_error"], "private")
        self.assertIsNotNone(self.db.get_subreddit("ok1")["last_fetched_at"])

    # ---- pre-fetch layer: discovery ----
    def test_mention_extraction(self):
        self.assertEqual(extract_mentions("try r/Options and /r/thetagang, not bar/optionsx or r/ab"),
                         {"options", "thetagang"})

    def test_discovery_search_snowball_seeds_and_hydrate(self):
        fake = FakeReddit()
        fake.search["trading"] = [sub_data("Forex", 50000, "forex traders"),
                                  sub_data("PokemonTrading", 90000, "trade pokemon cards")]
        fake.search["options"] = [sub_data("Forex", 50000, "forex traders")]        # same community, second query
        fake.about["algotrading"] = sub_data("algotrading", 120000, "Algo traders. See also r/quantfinance")
        fake.about["quantfinance"] = sub_data("quantfinance", 30000, "quant folks")
        fake.about["optionsmasters"] = sub_data("optionsmasters", 8000, "options strategies")
        self.db.upsert_raw_post("m1", "algotrading", raw("m1", "See r/optionsmasters for this"))
        self.db.upsert_raw_post("m2", "algotrading", raw("m2", "Other post", body="also try r/optionsmasters"))

        stats = run_discovery(self.db, fake, self.s, queries=["trading", "options"], seeds=["r/algotrading"],
                              hydrate_limit=10)
        self.assertEqual((stats["search_hits"], stats["new_from_search"]), (3, 2))   # Forex de-duplicated
        self.assertEqual(stats["seeds_added"], 1)
        self.assertEqual(self.db.get_subreddit("algotrading")["status"], "approved")
        self.assertEqual(self.db.get_subreddit("forex")["status"], "candidate")      # never auto-approved
        self.assertEqual(self.db.get_subreddit("forex")["subscribers"], 50000)
        self.assertEqual(self.db.get_subreddit("quantfinance")["discovered_via"], "sidebar:algotrading")
        opt = self.db.get_subreddit("optionsmasters")
        self.assertEqual((opt["discovered_via"], opt["mention_count"]), ("mention", 2))
        self.assertEqual(opt["subscribers"], 8000)                                   # hydrated via about.json
        self.assertEqual(stats["hydrated"], 2)

    def test_snowball_ignores_rarely_mentioned_names(self):
        self.db.upsert_raw_post("m1", "algotrading", raw("m1", "once: r/lonelymention"))
        run_discovery(self.db, FakeReddit(), self.s, queries=[], hydrate_limit=0)
        self.assertIsNone(self.db.get_subreddit("lonelymention"))

    # ---- pre-fetch layer: curation ----
    def test_curation_rules_classifier_review_and_fetch(self):
        fake = FakeReddit()
        cands = {
            "tinysub": sub_data("tinysub", 200, "options chat"),
            "pokemontrading": sub_data("pokemontrading", 90000, "trade pokemon cards"),
            "forex": sub_data("forex", 50000, "forex traders"),
            "gardenclub": sub_data("gardenclub", 40000, "growing plants"),
            "deadoptions": sub_data("deadoptions", 30000, "options trading"),
            "privatesub": sub_data("privatesub", 20000, "stocks talk"),
        }
        for name, meta in cands.items():
            self.db.upsert_subreddit(name, meta={"display_name": name, "title": meta["title"],
                                                 "description": meta["public_description"],
                                                 "subscribers": meta["subscribers"], "over18": 0,
                                                 "subreddit_type": "public", "quarantine": 0}, via="search:x")
        active = [raw(f"f{i}", f"title {i}", created=NOW - i * 3600, subreddit="forex") for i in range(25)]
        fake.listing.update({"forex": active, "gardenclub": active,
                             "deadoptions": [raw("d1", "old", created=NOW - 300 * 86400)],
                             "privatesub": SubredditUnavailable("private")})

        stats = classify_candidates(self.db, fake, StubLLM(), self.s)
        self.assertEqual(stats, {"rejected:too_small": 1, "rejected:off_topic_keywords": 1, "rejected:inactive": 1,
                                 "rejected:private": 1, "rejected:llm": 1, "pending_review": 1})
        self.assertEqual(self.db.get_subreddit("forex")["status"], "candidate")      # waits for a human

        path, n = export_review(self.db, self.s)
        self.assertEqual(n, 1)
        rows = read_csv(path)
        self.assertEqual(rows[0]["name"], "forex")
        rows[0]["decision"] = "approve"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        self.assertEqual(apply_review(self.db, path), {"approved": 1})
        self.assertEqual(self.db.get_subreddit("forex")["status"], "approved")
        self.assertEqual(self.db.get_subreddit("forex")["decided_by"], "manual")

        fake.listing["forex"] = [raw("fx1", "t", created=NOW - 100, subreddit="forex")]
        self.assertEqual(fetch_all(self.db, fake, self.s)["new_posts"], 1)           # fetches approved only

    def test_auto_approve_option(self):
        fake = FakeReddit()
        self.db.upsert_subreddit("forex", meta={"display_name": "forex", "title": "Forex", "description": "forex traders",
                                                "subscribers": 50000, "over18": 0, "subreddit_type": "public",
                                                "quarantine": 0}, via="search:x")
        fake.listing["forex"] = [raw(f"f{i}", "t", created=NOW - i * 3600, subreddit="forex") for i in range(25)]
        self.assertEqual(classify_candidates(self.db, fake, StubLLM(), self.s, auto_approve=True), {"approved:llm": 1})
        self.assertEqual(self.db.get_subreddit("forex")["status"], "approved")


if __name__ == "__main__":
    unittest.main()
