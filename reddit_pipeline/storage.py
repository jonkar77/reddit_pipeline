"""SQLite persistence. One file holds every layer so a run is fully reproducible and replayable.

Layers
  raw_posts / raw_comments   untouched API payloads (never re-fetch to reprocess)
  posts                      cleaned, normalized, deduped; `status` = eligible | filtered (+ reason)
  subreddits                 discovery layer: candidate/approved/rejected communities + metadata
  triage                     cheap title+body verdict per post (gates comments + full analysis)
  extractions                LLM output, cached by input hash (content + model + prompt version)
  failed_records             dead-letter queue, replayable
  runs / opportunities       clustering+scoring output, tied to the model/prompt that produced it
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_posts (
    reddit_id TEXT PRIMARY KEY, subreddit TEXT NOT NULL, payload_json TEXT NOT NULL,
    first_seen_at TEXT NOT NULL, fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_comments (
    post_id TEXT PRIMARY KEY, format TEXT NOT NULL,          -- 'api' (raw response) | 'flat' (imported)
    payload_json TEXT NOT NULL, fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fetch_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY, subreddit TEXT, title TEXT, selftext TEXT, flair TEXT, author_hash TEXT,
    score INTEGER, num_comments INTEGER, engagement INTEGER, created_utc REAL, permalink TEXT,
    content_hash TEXT, status TEXT NOT NULL, filter_reason TEXT, processed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status, engagement DESC);

CREATE TABLE IF NOT EXISTS subreddits (
    name TEXT PRIMARY KEY,                       -- lowercase key
    display_name TEXT, title TEXT, description TEXT, subscribers INTEGER, created_utc REAL,
    over18 INTEGER, subreddit_type TEXT, quarantine INTEGER,
    status TEXT NOT NULL DEFAULT 'candidate',    -- candidate | approved | rejected
    decided_by TEXT,                             -- seed | rule | llm | manual | import
    category TEXT, is_trading INTEGER, reason TEXT,
    discovered_via TEXT, mention_count INTEGER DEFAULT 0,
    posts_per_week REAL, last_activity_utc REAL, sample_titles_json TEXT,
    first_seen_at TEXT, metadata_at TEXT, classified_at TEXT, last_fetched_at TEXT, last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_subreddits_status ON subreddits(status);

CREATE TABLE IF NOT EXISTS triage (
    post_id TEXT PRIMARY KEY, input_hash TEXT, model TEXT, prompt_version TEXT,
    worth INTEGER NOT NULL, post_type TEXT, reason TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS extractions (
    input_hash TEXT PRIMARY KEY, post_id TEXT NOT NULL, model TEXT, prompt_version TEXT,
    result_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_extractions_post ON extractions(post_id);

CREATE TABLE IF NOT EXISTS failed_records (
    post_id TEXT NOT NULL, stage TEXT NOT NULL, input_hash TEXT, error TEXT,
    attempts INTEGER DEFAULT 1, last_failed_at TEXT, resolved INTEGER DEFAULT 0,
    PRIMARY KEY (post_id, stage)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, stage TEXT, started_at TEXT, finished_at TEXT,
    model TEXT, prompt_version TEXT, params_json TEXT, metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS opportunities (
    run_id TEXT NOT NULL, rank INTEGER NOT NULL,
    opportunity TEXT, pain_reason TEXT, target_user TEXT, mvp_idea TEXT,
    existing_alternatives TEXT, monetization TEXT,
    score REAL, num_posts INTEGER, unique_authors INTEGER, total_engagement INTEGER,
    avg_intensity REAL, breakdown_json TEXT, member_ids_json TEXT, evidence_links TEXT,
    num_communities INTEGER, communities TEXT,
    PRIMARY KEY (run_id, rank)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created (CREATE IF NOT EXISTS won't)."""
        wanted = {"opportunities": {"num_communities": "INTEGER", "communities": "TEXT"}}
        for table, cols in wanted.items():
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for col, typ in cols.items():
                if col not in have:
                    with self.conn:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    def close(self) -> None:
        self.conn.close()

    # ---- fetch checkpoints -------------------------------------------------
    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM fetch_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO fetch_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def delete_state(self, key: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM fetch_state WHERE key=?", (key,))

    # ---- raw layer ---------------------------------------------------------
    def upsert_raw_post(self, reddit_id: str, subreddit: str, payload: dict) -> bool:
        """Idempotent. Returns True if the post was new."""
        existed = self.conn.execute(
            "SELECT 1 FROM raw_posts WHERE reddit_id=?", (reddit_id,)).fetchone() is not None
        ts = now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO raw_posts(reddit_id, subreddit, payload_json, first_seen_at, fetched_at) "
                "VALUES(?,?,?,?,?) ON CONFLICT(reddit_id) DO UPDATE SET "
                "payload_json=excluded.payload_json, fetched_at=excluded.fetched_at",
                (reddit_id, subreddit, json.dumps(payload), ts, ts))
        return not existed

    def iter_raw_posts(self, subreddit: str | None = None) -> Iterator[tuple[str, dict]]:
        sql, args = "SELECT reddit_id, payload_json FROM raw_posts", ()
        if subreddit:
            sql, args = sql + " WHERE lower(subreddit)=lower(?)", (subreddit,)
        for row in self.conn.execute(sql, args):
            yield row["reddit_id"], json.loads(row["payload_json"])

    def iter_raw_comments(self) -> Iterator[tuple[str, object, str]]:
        for row in self.conn.execute("SELECT post_id, payload_json, format FROM raw_comments"):
            yield row["post_id"], json.loads(row["payload_json"]), row["format"]

    def save_comments(self, post_id: str, payload, fmt: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO raw_comments(post_id, format, payload_json, fetched_at) VALUES(?,?,?,?) "
                "ON CONFLICT(post_id) DO UPDATE SET format=excluded.format, "
                "payload_json=excluded.payload_json, fetched_at=excluded.fetched_at",
                (post_id, fmt, json.dumps(payload), now()))

    def get_comments(self, post_id: str):
        """Return (payload, format) or None if comments were never fetched."""
        row = self.conn.execute(
            "SELECT format, payload_json FROM raw_comments WHERE post_id=?", (post_id,)).fetchone()
        return (json.loads(row["payload_json"]), row["format"]) if row else None

    # ---- processed layer ---------------------------------------------------
    def save_posts(self, rows: list[dict]) -> None:
        cols = ["id", "subreddit", "title", "selftext", "flair", "author_hash", "score", "num_comments",
                "engagement", "created_utc", "permalink", "content_hash", "status", "filter_reason"]
        sql = (f"INSERT OR REPLACE INTO posts({', '.join(cols)}, processed_at) "
               f"VALUES({', '.join('?' for _ in cols)}, ?)")
        ts = now()
        with self.conn:
            self.conn.executemany(sql, [[r[c] for c in cols] + [ts] for r in rows])

    def median_engagements(self) -> dict[str, float]:
        """Median engagement per subreddit (lowercase key). Used to compare posts across communities."""
        by_sub: dict[str, list[int]] = {}
        for r in self.conn.execute("SELECT subreddit, engagement FROM posts WHERE status='eligible'"):
            by_sub.setdefault((r["subreddit"] or "").lower(), []).append(r["engagement"])
        return {k: max(1.0, float(statistics.median(v))) for k, v in by_sub.items()}

    def _rank_relative(self, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        """Best first, measured against each post's own community so big subs don't hog the budget."""
        med = self.median_engagements()
        return sorted(rows, key=lambda r: (-(r["engagement"] / med.get((r["subreddit"] or "").lower(), 1.0)), r["id"]))

    def eligible_posts(self, subreddit: str | None = None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM posts WHERE status='eligible'", []
        if subreddit:
            sql += " AND lower(subreddit)=lower(?)"
            args.append(subreddit)
        return self._rank_relative(self.conn.execute(sql, args).fetchall())

    def posts_needing_comments(self, subreddit: str = "", limit: int = 0, require_triage: bool = False,
                               triage_version: str = "") -> list[sqlite3.Row]:
        sql = ("SELECT p.* FROM posts p LEFT JOIN raw_comments c ON c.post_id = p.id "
               "WHERE p.status='eligible' AND p.num_comments > 0 AND c.post_id IS NULL")
        args: list = []
        if subreddit:
            sql += " AND lower(p.subreddit)=lower(?)"
            args.append(subreddit)
        if require_triage:
            sql += (" AND p.id IN (SELECT post_id FROM triage WHERE worth=1 AND prompt_version=?)")
            args.append(triage_version)
        rows = self._rank_relative(self.conn.execute(sql, args).fetchall())
        return rows[:limit] if limit else rows

    # ---- triage ---------------------------------------------------------------
    def get_triage_hash(self, post_id: str) -> str | None:
        row = self.conn.execute("SELECT input_hash FROM triage WHERE post_id=?", (post_id,)).fetchone()
        return row["input_hash"] if row else None

    def save_triage(self, post_id: str, input_hash: str, model: str, prompt_version: str,
                    worth: bool, post_type: str, reason: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO triage(post_id, input_hash, model, prompt_version, worth, post_type, "
                "reason, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (post_id, input_hash, model, prompt_version, int(worth), post_type, reason[:200], now()))

    def triage_map(self, prompt_version: str) -> dict[str, bool]:
        return {r["post_id"]: bool(r["worth"]) for r in self.conn.execute(
            "SELECT post_id, worth FROM triage WHERE prompt_version=?", (prompt_version,))}

    # ---- subreddit registry (discovery layer) ------------------------------------
    META_COLS = ("display_name", "title", "description", "subscribers", "created_utc", "over18",
                 "subreddit_type", "quarantine")
    SUB_COLS = META_COLS + ("status", "decided_by", "category", "is_trading", "reason", "discovered_via",
                            "mention_count", "posts_per_week", "last_activity_utc", "sample_titles_json",
                            "metadata_at", "classified_at", "last_fetched_at", "last_error")

    def get_subreddit(self, name: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM subreddits WHERE name=?", (name.lower(),)).fetchone()

    def upsert_subreddit(self, name: str, *, meta: dict | None = None, via: str = "", status: str | None = None,
                         decided_by: str | None = None, force_status: bool = False, mentions: int = 0) -> bool:
        """Insert or refresh a community. Never downgrades an existing decision unless force_status=True.
        Returns True if the community was new."""
        key = name.lower()
        existing = self.get_subreddit(key)
        cols: dict = {}
        if meta:
            cols.update({c: meta.get(c) for c in self.META_COLS if c in meta})
            cols["metadata_at"] = now()
        if existing is None:
            cols.setdefault("display_name", name)
            cols.update(status=status or "candidate", decided_by=decided_by, discovered_via=via,
                        mention_count=mentions, first_seen_at=now())
            names = ["name"] + list(cols)
            with self.conn:
                self.conn.execute(
                    f"INSERT INTO subreddits({', '.join(names)}) VALUES({', '.join('?' for _ in names)})",
                    [key] + list(cols.values()))
            return True
        if mentions > (existing["mention_count"] or 0):
            cols["mention_count"] = mentions
        if force_status and status:
            cols.update(status=status, decided_by=decided_by)
        if cols:
            self.update_subreddit(key, **cols)
        return False

    def update_subreddit(self, name: str, **cols) -> None:
        bad = set(cols) - set(self.SUB_COLS)
        if bad:
            raise ValueError(f"unknown subreddit columns: {sorted(bad)}")
        with self.conn:
            self.conn.execute(
                f"UPDATE subreddits SET {', '.join(f'{c}=?' for c in cols)} WHERE name=?",
                list(cols.values()) + [name.lower()])

    def subreddits(self, status: str | None = None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM subreddits", []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        return self.conn.execute(sql + " ORDER BY COALESCE(subscribers, 0) DESC, name", args).fetchall()

    def approved_subreddits(self) -> list[sqlite3.Row]:
        return self.subreddits("approved")

    def subreddits_needing_metadata(self, limit: int = 0) -> list[sqlite3.Row]:
        sql = "SELECT * FROM subreddits WHERE status='candidate' AND metadata_at IS NULL ORDER BY mention_count DESC, name"
        return self.conn.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()

    def subreddits_to_classify(self, limit: int = 0) -> list[sqlite3.Row]:
        sql = ("SELECT * FROM subreddits WHERE status='candidate' AND metadata_at IS NOT NULL "
               "AND classified_at IS NULL ORDER BY COALESCE(subscribers, 0) DESC, name")
        return self.conn.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()

    def subreddit_status_map(self) -> dict[str, str]:
        return {r["name"]: r["status"] for r in self.conn.execute("SELECT name, status FROM subreddits")}

    # ---- extractions (LLM cache) -------------------------------------------
    def get_extraction(self, input_hash: str) -> dict | None:
        row = self.conn.execute(
            "SELECT result_json FROM extractions WHERE input_hash=?", (input_hash,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def save_extraction(self, input_hash: str, post_id: str, model: str, prompt_version: str,
                        result: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO extractions(input_hash, post_id, model, prompt_version, "
                "result_json, created_at) VALUES(?,?,?,?,?,?)",
                (input_hash, post_id, model, prompt_version, json.dumps(result), now()))

    def analysis_records(self, model: str | None = None, prompt_version: str | None = None,
                         subreddit: str | None = None) -> list[dict]:
        """Latest extraction per eligible post, merged with the post's metadata."""
        sql = ("SELECT e.result_json, e.created_at, p.id, p.subreddit, p.title, p.permalink, p.score, "
               "p.num_comments, p.engagement, p.author_hash, p.created_utc "
               "FROM extractions e JOIN posts p ON p.id = e.post_id WHERE p.status='eligible'")
        args: list = []
        for col, val in (("e.model", model), ("e.prompt_version", prompt_version),
                         ("lower(p.subreddit)", subreddit.lower() if subreddit else None)):
            if val:
                sql += f" AND {col} = ?"
                args.append(val)
        latest: dict[str, dict] = {}
        for row in self.conn.execute(sql + " ORDER BY e.created_at ASC", args):  # later rows win
            rec = {k: row[k] for k in row.keys() if k not in ("result_json", "created_at")}
            rec.update(json.loads(row["result_json"]))
            latest[row["id"]] = rec
        return list(latest.values())

    # ---- dead-letter queue -------------------------------------------------
    def record_failure(self, post_id: str, stage: str, input_hash: str | None, error: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO failed_records(post_id, stage, input_hash, error, attempts, last_failed_at, resolved) "
                "VALUES(?,?,?,?,1,?,0) ON CONFLICT(post_id, stage) DO UPDATE SET "
                "input_hash=excluded.input_hash, error=excluded.error, attempts=attempts+1, "
                "last_failed_at=excluded.last_failed_at, resolved=0",
                (post_id, stage, input_hash, error[:500], now()))

    def resolve_failure(self, post_id: str, stage: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE failed_records SET resolved=1 WHERE post_id=? AND stage=?", (post_id, stage))

    def unresolved_failures(self, stage: str | None = None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM failed_records WHERE resolved=0", []
        if stage:
            sql += " AND stage=?"
            args.append(stage)
        return self.conn.execute(sql, args).fetchall()

    # ---- runs & results ----------------------------------------------------
    def start_run(self, stage: str, model: str = "", prompt_version: str = "", params: dict | None = None) -> str:
        run_id = f"{stage}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs(run_id, stage, started_at, model, prompt_version, params_json) VALUES(?,?,?,?,?,?)",
                (run_id, stage, now(), model, prompt_version, json.dumps(params or {})))
        return run_id

    def finish_run(self, run_id: str, metrics: dict) -> None:
        with self.conn:
            self.conn.execute("UPDATE runs SET finished_at=?, metrics_json=? WHERE run_id=?",
                              (now(), json.dumps(metrics), run_id))

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def latest_run_id(self, stage: str) -> str | None:
        row = self.conn.execute(
            "SELECT run_id FROM runs WHERE stage=? AND finished_at IS NOT NULL ORDER BY started_at DESC LIMIT 1",
            (stage,)).fetchone()
        return row["run_id"] if row else None

    def save_opportunities(self, run_id: str, rows: list[dict]) -> None:
        cols = ["rank", "opportunity", "pain_reason", "target_user", "mvp_idea", "existing_alternatives",
                "monetization", "score", "num_posts", "unique_authors", "total_engagement",
                "avg_intensity", "breakdown_json", "member_ids_json", "evidence_links",
                "num_communities", "communities"]
        sql = (f"INSERT INTO opportunities(run_id, {', '.join(cols)}) "
               f"VALUES(?, {', '.join('?' for _ in cols)})")
        with self.conn:
            self.conn.executemany(sql, [[run_id] + [r[c] for c in cols] for r in rows])

    def get_opportunities(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM opportunities WHERE run_id=? ORDER BY rank", (run_id,))]

    # ---- reporting ---------------------------------------------------------
    def counts(self) -> dict:
        def n(sql: str, *a) -> int:
            return self.conn.execute(sql, a).fetchone()[0]
        return {
            "raw_posts": n("SELECT COUNT(*) FROM raw_posts"),
            "raw_comment_threads": n("SELECT COUNT(*) FROM raw_comments"),
            "posts_eligible": n("SELECT COUNT(*) FROM posts WHERE status='eligible'"),
            "posts_filtered": n("SELECT COUNT(*) FROM posts WHERE status='filtered'"),
            "subreddits_approved": n("SELECT COUNT(*) FROM subreddits WHERE status='approved'"),
            "subreddits_candidate": n("SELECT COUNT(*) FROM subreddits WHERE status='candidate'"),
            "subreddits_rejected": n("SELECT COUNT(*) FROM subreddits WHERE status='rejected'"),
            "triaged": n("SELECT COUNT(*) FROM triage"),
            "triage_passed": n("SELECT COUNT(*) FROM triage WHERE worth=1"),
            "extractions": n("SELECT COUNT(*) FROM extractions"),
            "failed_unresolved": n("SELECT COUNT(*) FROM failed_records WHERE resolved=0"),
            "opportunity_runs": n("SELECT COUNT(*) FROM runs WHERE stage='opportunities'"),
        }

    def filter_reasons(self) -> dict:
        return {r["filter_reason"]: r["c"] for r in self.conn.execute(
            "SELECT filter_reason, COUNT(*) c FROM posts WHERE status='filtered' GROUP BY 1")}
