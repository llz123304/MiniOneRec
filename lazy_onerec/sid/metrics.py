"""Metrics for clusters induced by complete semantic IDs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from tqdm.auto import tqdm


def _complete_sid_keys(
    codes: np.ndarray,
    codebook_sizes: Sequence[int],
) -> np.ndarray | None:
    """Pack complete SIDs into uint64 keys when the code space permits."""
    maximum = np.iinfo(np.uint64).max
    possible = 1
    for size in codebook_sizes:
        if possible > maximum // size:
            return None
        possible *= size

    keys = np.zeros(len(codes), dtype=np.uint64)
    for level, size in enumerate(codebook_sizes):
        keys = (
            keys * np.uint64(size)
            + codes[:, level].astype(np.uint64, copy=False)
        )
    return keys


def evaluate_final_sid_clusters(
    codes: np.ndarray,
    codebook_sizes: Sequence[int],
) -> Dict[str, Any]:
    """Evaluate cluster sizes formed by unique complete SID tuples."""
    values = np.asarray(codes)
    sizes = tuple(int(size) for size in codebook_sizes)
    if values.ndim != 2:
        raise ValueError(f"expected [N,L] SID codes, got {values.shape}")
    if len(values) == 0:
        raise ValueError("cannot evaluate empty SID codes")
    if values.shape[1] != len(sizes):
        raise ValueError(
            f"received {values.shape[1]} code columns for {len(sizes)} levels"
        )

    for level, size in enumerate(sizes):
        level_codes = values[:, level]
        if size <= 0:
            raise ValueError(f"codebook size at level {level} must be positive")
        if level_codes.min() < 0 or level_codes.max() >= size:
            raise ValueError(
                f"codes at level {level} must be in [0, {size})"
            )

    with tqdm(
        total=2,
        desc="Evaluating final SID clusters",
        unit="stage",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix_str("grouping complete SIDs")
        keys = _complete_sid_keys(values, sizes)
        if keys is None:
            _, cluster_sizes = np.unique(
                values,
                axis=0,
                return_counts=True,
            )
        else:
            _, cluster_sizes = np.unique(keys, return_counts=True)
        progress.update()

        progress.set_postfix_str("computing cluster statistics")
        n_items = int(len(values))
        n_clusters = int(len(cluster_sizes))
        singleton_clusters = int(np.count_nonzero(cluster_sizes == 1))
        probabilities = cluster_sizes.astype(np.float64) / n_items
        entropy = float(
            -(probabilities * np.log(probabilities)).sum()
        )
        normalized_entropy = (
            float(entropy / math.log(n_clusters))
            if n_clusters > 1
            else 0.0
        )
        quantiles = np.percentile(
            cluster_sizes,
            [50, 90, 95, 99],
            method="nearest",
        ).astype(np.int64)
        progress.update()

    return {
        "format": "semantic-id-cluster-metrics",
        "version": 1,
        "cluster_definition": "items sharing the same complete SID",
        "num_items": n_items,
        "num_clusters": n_clusters,
        "effective_cluster_count": n_clusters,
        "collision_count": int(n_items - n_clusters),
        "collision_rate": float(1.0 - n_clusters / n_items),
        "singleton_cluster_count": singleton_clusters,
        "singleton_cluster_ratio": float(singleton_clusters / n_clusters),
        "singleton_item_ratio": float(singleton_clusters / n_items),
        "max_cluster_size": int(cluster_sizes.max()),
        "mean_cluster_size": float(cluster_sizes.mean()),
        "cluster_size_std": float(cluster_sizes.std()),
        "sid_distribution_entropy_nats": entropy,
        "normalized_sid_distribution_entropy": normalized_entropy,
        "effective_cluster_perplexity": float(math.exp(entropy)),
        "quantile_population": "effective complete-SID clusters",
        "quantile_method": "nearest",
        "cluster_size_quantiles": {
            "p50": int(quantiles[0]),
            "p90": int(quantiles[1]),
            "p95": int(quantiles[2]),
            "p99": int(quantiles[3]),
        },
    }


def save_sid_metrics(metrics: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_sid_metrics(metrics: Dict[str, Any]) -> None:
    quantiles = metrics["cluster_size_quantiles"]
    print(
        "[sid-metrics] "
        f"clusters={metrics['num_clusters']} "
        f"collision_rate={metrics['collision_rate']:.6f} "
        f"singleton_ratio={metrics['singleton_cluster_ratio']:.6f} "
        f"entropy={metrics['sid_distribution_entropy_nats']:.6f} "
        f"normalized_entropy="
        f"{metrics['normalized_sid_distribution_entropy']:.6f} "
        f"mean_size={metrics['mean_cluster_size']:.2f} "
        f"max_size={metrics['max_cluster_size']} "
        f"p50={quantiles['p50']} p90={quantiles['p90']} "
        f"p95={quantiles['p95']} p99={quantiles['p99']}"
    )
