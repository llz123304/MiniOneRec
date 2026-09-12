"""Self-contained SID generation layer."""

from .artifact import SemanticIDArtifact
from .builders import (
    build_constrained_rq_kmeans,
    build_rq_kmeans,
    build_rq_kmeans_plus,
    build_rq_vae,
)
from .constrained_rq_kmeans import residual_kmeans
from .token_codec import BOS_ID, EOS_ID, PAD_ID, SidTokenCodec

__all__ = [
    "SemanticIDArtifact",
    "build_rq_kmeans",
    "build_constrained_rq_kmeans",
    "build_rq_vae",
    "build_rq_kmeans_plus",
    "residual_kmeans",
    "SidTokenCodec",
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
]
