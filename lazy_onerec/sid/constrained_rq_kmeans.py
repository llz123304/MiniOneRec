"""Standalone constrained residual K-means adapted from MiniOneRec.

This module contains only the SID algorithm. It has no assumptions about
Amazon, KuaiRand, model token ids, or training examples.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm


def balanced_kmeans(
    values: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-7,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from k_means_constrained import KMeansConstrained
    except ImportError as exc:
        raise ImportError(
            "Install k-means-constrained to build SIDs: "
            "pip install k-means-constrained"
        ) from exc

    n_items = values.shape[0]
    min_size = max(1, n_items // n_clusters - 1)
    max_size = n_items // n_clusters + 1
    model = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=min_size,
        size_max=max_size,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=3,
        n_jobs=16,
    )
    labels = model.fit_predict(values.astype(np.float32, copy=False))
    return labels.astype(np.int32), model.cluster_centers_.astype(np.float32)


def residual_kmeans(
    embeddings: np.ndarray,
    codebook_sizes: Sequence[int],
    max_iter: int = 100,
    tol: float = 1e-7,
    seed: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Return ``codes [N,L]``, codebooks, and reconstructed embeddings."""
    values = embeddings.astype(np.float32, copy=False)
    residual = values.copy()
    codes = np.empty((values.shape[0], len(codebook_sizes)), dtype=np.int32)
    codebooks: List[np.ndarray] = []

    levels = tqdm(
        enumerate(codebook_sizes),
        total=len(codebook_sizes),
        desc="Constrained RQ-Kmeans",
        unit="level",
        dynamic_ncols=True,
    )
    for level, size in levels:
        levels.set_postfix(level=level + 1, clusters=int(size))
        level_seed = int(np.random.RandomState(seed + level).randint(0, 2**31 - 1))
        level_codes, centroids = balanced_kmeans(
            residual,
            n_clusters=int(size),
            max_iter=max_iter,
            tol=tol,
            random_state=level_seed,
        )
        codes[:, level] = level_codes
        codebooks.append(centroids)
        residual -= centroids[level_codes]

    return codes, codebooks, values - residual
