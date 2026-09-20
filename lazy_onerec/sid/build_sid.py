"""Unified CLI for all supported MiniOneRec SID construction methods."""

from __future__ import annotations

import argparse

from .builders import (
    build_constrained_rq_kmeans,
    build_onerec_balanced_kmeans,
    build_rq_kmeans,
    build_rq_kmeans_plus,
    build_rq_vae,
)
from .distance import DISTANCE_METRICS


METHODS = (
    "rq-kmeans",
    "balanced-kmeans",
    "constrained-rq-kmeans",
    "rq-vae",
    "rq-kmeans-plus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--item-ids")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--codebook-sizes", type=int, nargs="+", default=[512, 512, 512]
    )
    parser.add_argument(
        "--distance-metric",
        choices=DISTANCE_METRICS,
        default="cosine",
    )
    parser.add_argument("--require-unique", action="store_true")

    # K-means backends
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument(
        "--distance-mode",
        choices=("auto", "matrix", "streaming"),
        default="auto",
    )
    parser.add_argument("--distance-batch-size", type=int, default=65536)

    # Neural backends
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="*", default=[512, 256, 128]
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--quant-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-kmeans-init", action="store_true")
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument(
        "--sinkhorn-epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0]
    )
    parser.add_argument("--sinkhorn-iters", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = dict(
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        codebook_sizes=args.codebook_sizes,
        distance_metric=args.distance_metric,
        item_ids_path=args.item_ids,
        require_unique=args.require_unique,
    )

    if args.method == "rq-kmeans":
        artifact = build_rq_kmeans(
            **common,
            beam_size=args.beam_size,
        )
    elif args.method == "constrained-rq-kmeans":
        artifact = build_constrained_rq_kmeans(
            **common,
            max_iter=args.max_iter,
            seed=args.seed,
        )
    elif args.method == "balanced-kmeans":
        artifact = build_onerec_balanced_kmeans(
            **common,
            max_iter=args.max_iter,
            seed=args.seed,
            device=args.device,
            distance_mode=args.distance_mode,
            distance_batch_size=args.distance_batch_size,
        )
    else:
        if len(args.sinkhorn_epsilons) != len(args.codebook_sizes):
            raise ValueError("one Sinkhorn epsilon is required per SID level")
        learning_rate = args.learning_rate
        if learning_rate is None:
            learning_rate = (
                1e-3 if args.method == "rq-vae" else 1e-4
            )
        neural = dict(
            **common,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dims,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            beta=args.beta,
            quant_loss_weight=args.quant_loss_weight,
            kmeans_init=not args.no_kmeans_init,
            kmeans_iters=args.kmeans_iters,
            sinkhorn_epsilons=args.sinkhorn_epsilons,
            sinkhorn_iters=args.sinkhorn_iters,
            eval_every=args.eval_every,
            seed=args.seed,
        )
        artifact = (
            build_rq_vae(**neural)
            if args.method == "rq-vae"
            else build_rq_kmeans_plus(**neural)
        )

    print(
        f"method={args.method} items={artifact.n_items} "
        f"sizes={list(artifact.codebook_sizes)} "
        f"collision_rate={artifact.collision_rate:.6f}"
    )


if __name__ == "__main__":
    main()
