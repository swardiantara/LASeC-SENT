"""Sentence embedding + clustering (the LASeC abstraction step).

Supported algorithms (settings follow the original LASeC grid):
  agglomerative  AgglomerativeClustering on cosine distances, `linkage` (default average), distance_threshold = t
  birch          Birch(threshold = t, n_clusters=None) on L2-normalised vectors (t is a Euclidean sub-cluster radius)
  dbscan         DBSCAN(eps = t, min_samples=1, metric=cosine)
  hdbscan        HDBSCAN(min_cluster_size=2, cluster_selection_epsilon = t) on cosine distances
  optics         OPTICS(max_eps = t, min_samples=2, metric=cosine)
For the density-based algorithms, points labelled as noise (-1) become singleton events: in triage a
sentence that matches nothing else is its own event.
"""
from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, AgglomerativeClustering, Birch
from sklearn.metrics.pairwise import cosine_distances

ALGORITHMS = ("agglomerative", "birch", "dbscan", "hdbscan", "optics")
ALGORITHM_LABEL = {"agglomerative": "Agglomerative", "birch": "BIRCH", "dbscan": "DBSCAN", "hdbscan": "HDBSCAN",
                   "optics": "OPTICS"}


def load_encoder(model: str, device: str | None = None, revision: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model, device=device, revision=revision)


def embed(encoder, sentences: Sequence[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    return encoder.encode(list(sentences), batch_size=batch_size, normalize_embeddings=normalize,
                          convert_to_numpy=True, show_progress_bar=False)


def _noise_to_singletons(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    noise = labels < 0
    labels[noise] = labels.max(initial=-1) + 1 + np.arange(noise.sum())
    return labels


def cluster_labels(X: np.ndarray, algorithm: str, threshold: float, linkage: str = "average",
                   branching_factor: int = 50) -> np.ndarray:
    if len(X) == 1:
        return np.zeros(1, dtype=int)
    if algorithm == "agglomerative":
        if linkage == "ward":
            m = AgglomerativeClustering(n_clusters=None, linkage="ward", distance_threshold=threshold)
            return m.fit_predict(X)
        D = cosine_distances(X)
        m = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage=linkage, distance_threshold=threshold)
        return m.fit_predict(D)
    if algorithm == "birch":
        return Birch(threshold=threshold, n_clusters=None, branching_factor=branching_factor).fit_predict(X)
    if algorithm == "dbscan":
        return _noise_to_singletons(DBSCAN(eps=max(threshold, 1e-9), min_samples=1, metric="cosine").fit_predict(X))
    if algorithm == "hdbscan":
        D = np.clip(cosine_distances(X), 0, None)
        m = HDBSCAN(min_cluster_size=2, metric="precomputed", cluster_selection_epsilon=threshold, copy=True)
        return _noise_to_singletons(m.fit_predict(D))
    if algorithm == "optics":
        return _noise_to_singletons(OPTICS(max_eps=threshold, min_samples=2, metric="cosine").fit_predict(X))
    raise ValueError(f"unknown clustering algorithm {algorithm!r}; choose from {ALGORITHMS}")


def cluster_sentences(sentences: Sequence[str], X: np.ndarray, threshold: float, algorithm: str = "agglomerative",
                      order: str = "sorted", seed: int = 0, linkage: str = "average", branching_factor: int = 50) -> dict:
    """Cluster unique sentences; returns {sentence: cluster_id}.

    order='sorted' (default) feeds sentences alphabetically, order='shuffle' uses a seeded permutation.
    Agglomerative clustering is invariant to input order (up to exact distance ties); BIRCH is not.
    """
    idx = np.arange(len(sentences))
    if order == "sorted":
        idx = np.array(sorted(idx, key=lambda i: sentences[i]))
    elif order == "shuffle":
        idx = np.random.default_rng(seed).permutation(idx)
    labels = cluster_labels(X[idx], algorithm, threshold, linkage, branching_factor)
    return {sentences[i]: int(c) for i, c in zip(idx, labels)}


def threshold_grid(spec: dict) -> Iterable[float]:
    return [round(float(t), 4) for t in np.arange(spec["start"], spec["stop"] + 1e-9, spec["step"])]
