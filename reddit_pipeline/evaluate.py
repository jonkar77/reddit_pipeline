"""Evaluation loop: hand-label a random sample, then score any prompt/model change against it."""
from __future__ import annotations

import csv
import random
from pathlib import Path

from .config import Settings
from .storage import Database

TRUE, FALSE = {"1", "true", "yes", "y"}, {"0", "false", "no", "n"}


def make_sample(db: Database, s: Settings, n: int, path: str, seed: int = 7) -> int:
    """Write a random sample of analysed posts with blank label columns for you to fill in."""
    records = db.analysis_records(s.model or None, s.prompt_version, s.subreddit or None)
    random.Random(seed).shuffle(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["post_id", "title", "permalink", "pred_post_type", "pred_has_pain",
                    "label_has_pain", "label_post_type"])
        for r in records[:n]:
            w.writerow([r["id"], r["title"], "https://www.reddit.com" + r["permalink"],
                        r["post_type"], int(r["has_pain"]), "", ""])
    return min(n, len(records))


def score_labels(db: Database, s: Settings, labels_path: str) -> dict:
    """Compare current extractions against your labels (has_pain precision/recall/F1, post_type accuracy)."""
    preds = {r["id"]: r for r in db.analysis_records(s.model or None, s.prompt_version, s.subreddit or None)}
    tp = fp = fn = tn = 0
    type_ok = type_total = 0
    with open(labels_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pred = preds.get(row["post_id"])
            if not pred:
                continue
            lab = (row.get("label_has_pain") or "").strip().lower()
            if lab in TRUE | FALSE:
                truth = lab in TRUE
                tp += truth and pred["has_pain"]
                fp += (not truth) and pred["has_pain"]
                fn += truth and not pred["has_pain"]
                tn += (not truth) and not pred["has_pain"]
            lt = (row.get("label_post_type") or "").strip()
            if lt:
                type_total += 1
                type_ok += lt == pred["post_type"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"labeled": tp + fp + fn + tn, "precision": round(precision, 3), "recall": round(recall, 3),
               "f1": round(f1, 3), "post_type_accuracy": round(type_ok / type_total, 3) if type_total else None}
    run_id = db.start_run("eval", s.model, s.prompt_version, {"labels": labels_path})
    db.finish_run(run_id, metrics)       # logged so you can compare prompt versions over time
    return metrics
