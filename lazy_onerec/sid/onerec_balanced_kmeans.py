"""GPU-capable greedy Balanced K-means from the OneRec SID algorithm."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from .distance import validate_distance_metric


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Install PyTorch to use OneRec Balanced K-means"
        ) from exc
    return torch


def _resolve_device(requested: str):
    torch = _torch()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def _cluster_quotas(n_items: int, n_clusters: int) -> List[int]:
    if n_clusters <= 0 or n_items < n_clusters:
        raise ValueError("Balanced K-means requires N >= K > 0")
    base, remainder = divmod(n_items, n_clusters)
    return [
        base + (1 if cluster < remainder else 0)
        for cluster in range(n_clusters)
    ]


def _distance_to_center(
    values,
    center,
    metric: str,
    value_norms,
):
    torch = _torch()
    if metric == "cosine":
        center = torch.nn.functional.normalize(center, dim=0)
        similarities = values @ center
        similarities /= value_norms.clamp_min(1e-12)
        return 1.0 - similarities
    return (
        values.square().sum(dim=1)
        + center.square().sum()
        - 2.0 * values @ center
    ).clamp_min_(0)


def _distance_matrix(
    values,
    centers,
    metric: str,
    batch_size: int,
):
    torch = _torch()
    distances = torch.empty(
        (len(values), len(centers)),
        dtype=torch.float32,
        device=values.device,
    )
    if metric == "cosine":
        distance_centers = torch.nn.functional.normalize(centers, dim=1)
        value_norms = values.norm(dim=1, keepdim=True).clamp_min_(1e-12)
    else:
        distance_centers = centers
        center_norms = centers.square().sum(dim=1).unsqueeze(0)

    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        batch = values[start:end]
        if metric == "cosine":
            distances[start:end] = 1.0 - (
                batch @ distance_centers.t()
            ) / value_norms[start:end]
        else:
            distances[start:end] = (
                batch.square().sum(dim=1, keepdim=True)
                + center_norms
                - 2.0 * batch @ distance_centers.t()
            ).clamp_min_(0)
    return distances


def _streaming_nearest(
    values,
    center,
    unassigned,
    quota: int,
    metric: str,
    batch_size: int,
    value_norms,
):
    torch = _torch()
    best_distances = torch.empty(
        0,
        dtype=torch.float32,
        device=values.device,
    )
    best_indices = torch.empty(
        0,
        dtype=torch.long,
        device=values.device,
    )
    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        local = torch.nonzero(
            unassigned[start:end],
            as_tuple=False,
        ).flatten()
        if not len(local):
            continue
        global_indices = local + start
        distances = _distance_to_center(
            values[global_indices],
            center,
            metric,
            value_norms[global_indices],
        )
        local_quota = min(quota, len(distances))
        local_distances, positions = torch.topk(
            distances,
            k=local_quota,
            largest=False,
            sorted=False,
        )
        candidate_distances = torch.cat(
            (best_distances, local_distances)
        )
        candidate_indices = torch.cat(
            (best_indices, global_indices[positions])
        )
        keep = min(quota, len(candidate_distances))
        best_distances, positions = torch.topk(
            candidate_distances,
            k=keep,
            largest=False,
            sorted=False,
        )
        best_indices = candidate_indices[positions]
    if len(best_indices) != quota:
        raise RuntimeError(
            f"found {len(best_indices)} candidates for quota {quota}"
        )
    return best_indices


def _choose_distance_mode(
    requested: str,
    values,
    n_clusters: int,
) -> str:
    if requested not in {"auto", "matrix", "streaming"}:
        raise ValueError(
            "distance_mode must be 'auto', 'matrix', or 'streaming'"
        )
    if requested != "auto":
        return requested

    required = len(values) * n_clusters * np.dtype(np.float32).itemsize
    if values.device.type == "cuda":
        torch = _torch()
        free_bytes, _ = torch.cuda.mem_get_info(values.device)
        return "matrix" if required <= int(free_bytes * 0.6) else "streaming"
    return "matrix" if required <= 512 * 2**20 else "streaming"


def balanced_kmeans(
    values: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 20,
    seed: int = 42,
    distance_metric: str = "euclidean",
    device: str = "cuda",
    distance_mode: str = "auto",
    distance_batch_size: int = 65536,
    description: str = "OneRec Balanced K-means",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Greedily assign each centroid a fixed quota until labels converge."""
    torch = _torch()
    metric = validate_distance_metric(distance_metric)
    target_device = _resolve_device(device)
    if max_iter <= 0 or distance_batch_size <= 0:
        raise ValueError("iteration and distance batch sizes must be positive")

    if isinstance(values, torch.Tensor):
        tensor = values.to(
            device=target_device,
            dtype=torch.float32,
        ).contiguous()
    else:
        source = np.ascontiguousarray(values, dtype=np.float32)
        tensor = torch.from_numpy(source).to(target_device)
    if target_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    quotas = _cluster_quotas(len(tensor), n_clusters)
    rng = np.random.RandomState(seed)
    initial = rng.choice(len(tensor), size=n_clusters, replace=False)
    centers = tensor[
        torch.as_tensor(initial, device=target_device)
    ].clone()
    if metric == "cosine":
        centers = torch.nn.functional.normalize(centers, dim=1)
    mode = _choose_distance_mode(distance_mode, tensor, n_clusters)
    value_norms = (
        tensor.norm(dim=1).clamp_min_(1e-12)
        if metric == "cosine"
        else None
    )
    previous_labels = None
    converged = False
    iterations = 0

    progress = tqdm(
        range(max_iter),
        desc=f"{description} ({target_device}, {mode})",
        unit="iter",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for iteration in progress:
            iterations = iteration + 1
            distances = None
            if mode == "matrix":
                try:
                    distances = _distance_matrix(
                        tensor,
                        centers,
                        metric,
                        distance_batch_size,
                    )
                except torch.OutOfMemoryError:
                    if distance_mode != "auto":
                        raise
                    mode = "streaming"
                    torch.cuda.empty_cache()
                    progress.set_description(
                        f"{description} ({target_device}, {mode})"
                    )
            unassigned = torch.ones(
                len(tensor),
                dtype=torch.bool,
                device=target_device,
            )
            labels = torch.full(
                (len(tensor),),
                -1,
                dtype=torch.long,
                device=target_device,
            )
            for cluster, quota in enumerate(quotas):
                if cluster == n_clusters - 1:
                    selected = torch.nonzero(
                        unassigned,
                        as_tuple=False,
                    ).flatten()
                elif distances is not None:
                    cluster_distances = distances[:, cluster]
                    cluster_distances.masked_fill_(
                        ~unassigned,
                        float("inf"),
                    )
                    selected = torch.topk(
                        cluster_distances,
                        k=quota,
                        largest=False,
                        sorted=False,
                    ).indices
                else:
                    selected = _streaming_nearest(
                        tensor,
                        centers[cluster],
                        unassigned,
                        quota,
                        metric,
                        distance_batch_size,
                        value_norms,
                    )
                labels[selected] = cluster
                unassigned[selected] = False
                centers[cluster] = tensor[selected].mean(dim=0)
                if metric == "cosine":
                    centers[cluster] = torch.nn.functional.normalize(
                        centers[cluster],
                        dim=0,
                    )

            if bool(unassigned.any()) or bool((labels < 0).any()):
                raise RuntimeError("Balanced K-means left items unassigned")
            if previous_labels is not None and torch.equal(
                labels,
                previous_labels,
            ):
                converged = True
                break
            previous_labels = labels

    counts = torch.bincount(labels, minlength=n_clusters)
    expected = torch.as_tensor(
        quotas,
        dtype=counts.dtype,
        device=counts.device,
    )
    if not torch.equal(counts, expected):
        raise RuntimeError("Balanced K-means violated cluster quotas")
    return (
        labels.cpu().numpy().astype(np.int32, copy=False),
        centers.cpu().numpy().astype(np.float32, copy=False),
        {
            "device": str(target_device),
            "distance_mode": mode,
            "iterations": iterations,
            "converged": converged,
            "min_cluster_size": min(quotas),
            "max_cluster_size": max(quotas),
        },
    )


def residual_kmeans(
    embeddings: np.ndarray,
    codebook_sizes: Sequence[int],
    *,
    max_iter: int = 20,
    seed: int = 42,
    distance_metric: str = "euclidean",
    device: str = "cuda",
    distance_mode: str = "auto",
    distance_batch_size: int = 65536,
) -> Tuple[np.ndarray, List[np.ndarray], float, List[Dict[str, object]]]:
    """Run OneRec Balanced K-means independently at each residual level."""
    torch = _torch()
    values = np.ascontiguousarray(embeddings, dtype=np.float32)
    target_device = _resolve_device(device)
    residual = torch.from_numpy(values).to(target_device)
    codes = np.empty(
        (len(values), len(codebook_sizes)),
        dtype=np.int32,
    )
    codebooks: List[np.ndarray] = []
    level_stats: List[Dict[str, object]] = []

    for level, size in enumerate(codebook_sizes):
        level_codes, centers, stats = balanced_kmeans(
            residual,
            int(size),
            max_iter=max_iter,
            seed=seed + level,
            distance_metric=distance_metric,
            device=str(target_device),
            distance_mode=distance_mode,
            distance_batch_size=distance_batch_size,
            description=f"OneRec Balanced RQ level {level + 1}",
        )
        labels = torch.from_numpy(level_codes).to(target_device)
        center_tensor = torch.from_numpy(centers).to(target_device)
        codes[:, level] = level_codes
        codebooks.append(centers)
        level_stats.append(stats)
        with torch.no_grad():
            for start in range(0, len(residual), distance_batch_size):
                end = min(start + distance_batch_size, len(residual))
                residual[start:end].sub_(
                    center_tensor[labels[start:end]]
                )

    reconstruction_mse = float(residual.square().mean().cpu())
    return codes, codebooks, reconstruction_mse, level_stats
