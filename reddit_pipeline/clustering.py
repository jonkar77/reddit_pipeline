"""Stage 4a: group similar pains. Embeddings + greedy centroid (default) or HDBSCAN (optional)."""
from __future__ import annotations

import logging

import numpy as np

from .llm import Embedder

log = logging.getLogger(__name__)


def record_text(r: dict) -> str:
    # embed the PAIN (not the solution) so clusters mean "same problem", not "same product idea"
    return f"{r['pain_theme']}. {r['pain_summary']}"


def greedy_groups(vecs: np.ndarray, threshold: float) -> list[list[int]]:
    """Order-dependent: callers sort by engagement so strong posts seed the clusters."""
    centroids: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    members: list[list[int]] = []
    for idx, v in enumerate(vecs):
        if centroids:
            sims = np.array([float(v @ c) for c in centroids])
            best = int(sims.argmax())
            if sims[best] >= threshold:
                members[best].append(idx)
                sums[best] = sums[best] + v
                centroids[best] = sums[best] / (np.linalg.norm(sums[best]) + 1e-9)
                continue
        centroids.append(v.copy())
        sums.append(v.copy())
        members.append([idx])
    return members


def hdbscan_groups(vecs: np.ndarray, min_cluster_size: int = 2) -> list[list[int]]:
    """Noise points become singleton groups (later dropped by --min-authors), never silently lost."""
    if len(vecs) <= min_cluster_size:
        return [[i] for i in range(len(vecs))]
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError("cluster method 'hdbscan' needs scikit-learn: pip install scikit-learn") from exc
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(vecs)
    groups: dict[int, list[int]] = {}
    noise: list[list[int]] = []
    for i, label in enumerate(labels):
        if label == -1:
            noise.append([i])
        else:
            groups.setdefault(int(label), []).append(i)
    return list(groups.values()) + noise


def cluster_records(records: list[dict], embedder: Embedder | None, method: str,
                    threshold: float) -> list[list[dict]]:
    if embedder is not None:
        try:
            vecs = embedder.embed([record_text(r) for r in records])
            idx = hdbscan_groups(vecs) if method == "hdbscan" else greedy_groups(vecs, threshold)
            return [[records[i] for i in g] for g in idx]
        except RuntimeError:
            raise
        except Exception as exc:
            log.warning("Embedding failed (%s); falling back to grouping by identical pain_theme", exc)
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["pain_theme"].strip().lower(), []).append(r)
    return list(groups.values())
