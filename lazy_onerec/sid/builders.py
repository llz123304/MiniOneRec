"""Unified programmatic builders for all supported SID methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .artifact import SemanticIDArtifact
from .constrained_rq_kmeans import residual_kmeans
from .distance import prepare_numpy_values, validate_distance_metric
from .faiss_rq_kmeans import faiss_residual_kmeans
from .metrics import (
    evaluate_final_sid_clusters,
    print_sid_metrics,
    save_sid_metrics,
)
from .neural_rq import (
    build_rqvae_class,
    make_rq_kmeans_plus,
    train_neural_rq,
)


def load_embeddings(path: str) -> np.ndarray:
    values = np.load(path).astype(np.float32, copy=False)
    if values.ndim != 2:
        raise ValueError(f"expected [N,D] embeddings, got {values.shape}")
    return values


def load_item_ids(path: str | None, n_items: int) -> List[str]:
    if path is None:
        return [str(i) for i in range(n_items)]
    source = Path(path)
    if source.suffix == ".npy":
        values = np.load(source, allow_pickle=False).reshape(-1).tolist()
    elif source.suffix == ".json":
        with source.open(encoding="utf-8") as f:
            values = json.load(f)
        if not isinstance(values, list):
            raise ValueError("item-id JSON must be an ordered list")
    else:
        with source.open(encoding="utf-8") as f:
            values = [line.strip() for line in f if line.strip()]
    if len(values) != n_items:
        raise ValueError(f"{len(values)} item ids for {n_items} embeddings")
    return [str(value) for value in values]


def _save_outputs(
    output_dir: str,
    item_ids: Sequence[str],
    codes: np.ndarray,
    codebook_sizes: Sequence[int],
    codebooks: Sequence[np.ndarray],
    metadata: Dict[str, Any],
    require_unique: bool,
) -> SemanticIDArtifact:
    metrics = evaluate_final_sid_clusters(codes, codebook_sizes)
    metrics["method"] = metadata["method"]
    metrics["distance_metric"] = metadata["distance_metric"]
    metrics["codebook_sizes"] = [int(size) for size in codebook_sizes]
    if require_unique and metrics["collision_count"]:
        raise ValueError(
            f"artifact contains {metrics['collision_count']} collided items"
        )
    metadata = dict(metadata)
    metadata["metrics_file"] = "sid_metrics.json"
    metadata["sid_metrics_summary"] = {
        key: metrics[key]
        for key in (
            "num_items",
            "effective_cluster_count",
            "collision_count",
            "collision_rate",
        )
    }
    artifact = SemanticIDArtifact.from_rows(
        item_ids=item_ids,
        codes=codes,
        codebook_sizes=codebook_sizes,
        metadata=metadata,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact.save(output / "sid_index.json")
    np.save(output / "codes.npy", codes.astype(np.int32, copy=False))
    np.savez_compressed(
        output / "codebooks.npz",
        **{
            f"codebook_{level}": values
            for level, values in enumerate(codebooks)
        },
    )
    save_sid_metrics(metrics, output / "sid_metrics.json")
    print_sid_metrics(metrics)
    return artifact


def build_rq_kmeans(
    embeddings_path: str,
    output_dir: str,
    codebook_sizes: Sequence[int],
    item_ids_path: str | None = None,
    beam_size: int = 1,
    distance_metric: str = "euclidean",
    require_unique: bool = False,
) -> SemanticIDArtifact:
    distance_metric = validate_distance_metric(distance_metric)
    embeddings = prepare_numpy_values(
        load_embeddings(embeddings_path), distance_metric, copy=False
    )
    item_ids = load_item_ids(item_ids_path, len(embeddings))
    codes, codebooks, _ = faiss_residual_kmeans(
        embeddings,
        codebook_sizes,
        beam_size=beam_size,
        distance_metric=distance_metric,
    )
    return _save_outputs(
        output_dir,
        item_ids,
        codes,
        codebook_sizes,
        codebooks,
        {
            "method": "MiniOneRec-RQ-Kmeans",
            "embeddings_path": embeddings_path,
            "embedding_dim": int(embeddings.shape[1]),
            "beam_size": beam_size,
            "distance_metric": distance_metric,
        },
        require_unique,
    )


def build_constrained_rq_kmeans(
    embeddings_path: str,
    output_dir: str,
    codebook_sizes: Sequence[int],
    item_ids_path: str | None = None,
    max_iter: int = 100,
    seed: int = 42,
    distance_metric: str = "euclidean",
    require_unique: bool = False,
) -> SemanticIDArtifact:
    distance_metric = validate_distance_metric(distance_metric)
    embeddings = prepare_numpy_values(
        load_embeddings(embeddings_path), distance_metric, copy=False
    )
    item_ids = load_item_ids(item_ids_path, len(embeddings))
    codes, codebooks, reconstruction = residual_kmeans(
        embeddings,
        codebook_sizes,
        max_iter=max_iter,
        seed=seed,
        distance_metric=distance_metric,
    )
    return _save_outputs(
        output_dir,
        item_ids,
        codes,
        codebook_sizes,
        codebooks,
        {
            "method": "MiniOneRec-constrained-RQ-Kmeans",
            "embeddings_path": embeddings_path,
            "embedding_dim": int(embeddings.shape[1]),
            "seed": seed,
            "max_iter": max_iter,
            "distance_metric": distance_metric,
            "reconstruction_mse": float(
                np.mean((embeddings - reconstruction) ** 2)
            ),
        },
        require_unique,
    )


def _build_neural(
    method: str,
    embeddings_path: str,
    output_dir: str,
    codebook_sizes: Sequence[int],
    item_ids_path: str | None,
    latent_dim: int,
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    beta: float,
    quant_loss_weight: float,
    kmeans_init: bool,
    kmeans_iters: int,
    sinkhorn_epsilons: Sequence[float],
    sinkhorn_iters: int,
    eval_every: int,
    seed: int,
    distance_metric: str,
    require_unique: bool,
) -> SemanticIDArtifact:
    torch = __import__("torch")
    distance_metric = validate_distance_metric(distance_metric)
    embeddings = prepare_numpy_values(
        load_embeddings(embeddings_path), distance_metric, copy=False
    )
    item_ids = load_item_ids(item_ids_path, len(embeddings))
    rqvae_class = build_rqvae_class()

    if method == "rq-kmeans-plus":
        latent_dim = embeddings.shape[1]
        initial_codes, initial_codebooks, _ = residual_kmeans(
            embeddings,
            codebook_sizes,
            max_iter=kmeans_iters,
            seed=seed,
            distance_metric=distance_metric,
        )
        del initial_codes
        kmeans_init = False
    else:
        initial_codebooks = None

    model = rqvae_class(
        input_dim=embeddings.shape[1],
        codebook_sizes=codebook_sizes,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta,
        quant_loss_weight=quant_loss_weight,
        kmeans_init=kmeans_init,
        kmeans_iters=kmeans_iters,
        sinkhorn_epsilons=sinkhorn_epsilons,
        sinkhorn_iters=sinkhorn_iters,
        distance_metric=distance_metric,
    )
    if initial_codebooks is not None:
        make_rq_kmeans_plus(model, initial_codebooks)

    result = train_neural_rq(
        embeddings,
        model,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_every=eval_every,
        seed=seed,
    )
    artifact = _save_outputs(
        output_dir,
        item_ids,
        result.codes,
        codebook_sizes,
        result.codebooks,
        {
            "method": (
                "MiniOneRec-RQ-VAE"
                if method == "rq-vae"
                else "MiniOneRec-RQ-Kmeans+"
            ),
            "embeddings_path": embeddings_path,
            "embedding_dim": int(embeddings.shape[1]),
            "latent_dim": latent_dim,
            "hidden_dims": list(hidden_dims),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_loss": result.best_loss,
            "best_collision_rate": result.collision_rate,
            "distance_metric": distance_metric,
            "seed": seed,
        },
        require_unique,
    )
    output = Path(output_dir)
    torch.save(
        {
            "method": method,
            "state_dict": result.model.state_dict(),
            "input_dim": int(embeddings.shape[1]),
            "latent_dim": latent_dim,
            "hidden_dims": list(hidden_dims),
            "codebook_sizes": list(codebook_sizes),
            "distance_metric": distance_metric,
        },
        output / "model.pt",
    )
    if initial_codebooks is not None:
        np.savez_compressed(
            output / "initial_codebooks.npz",
            **{
                f"codebook_{level}": values
                for level, values in enumerate(initial_codebooks)
            },
        )
    return artifact


def build_rq_vae(**kwargs) -> SemanticIDArtifact:
    return _build_neural(method="rq-vae", **kwargs)


def build_rq_kmeans_plus(**kwargs) -> SemanticIDArtifact:
    return _build_neural(method="rq-kmeans-plus", **kwargs)
