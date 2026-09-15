"""SID-constrained ranking structures and inference metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch


EVALUATION_TOP_K = 10


class SidPrefixIndex:
    """Store valid next codes for every observed SID prefix."""

    def __init__(
        self,
        codes: Iterable[Sequence[int]],
        codebook_sizes: Sequence[int],
    ):
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        if not self.codebook_sizes:
            raise ValueError("codebook_sizes must not be empty")

        allowed_sets = [
            defaultdict(set) for _ in range(len(self.codebook_sizes))
        ]
        n_codes = 0
        for row in codes:
            normalized = tuple(int(code) for code in row)
            if len(normalized) != len(self.codebook_sizes):
                raise ValueError(
                    f"SID has {len(normalized)} levels; expected "
                    f"{len(self.codebook_sizes)}"
                )
            for level, (code, size) in enumerate(
                zip(normalized, self.codebook_sizes)
            ):
                if not 0 <= code < size:
                    raise ValueError(
                        f"level {level} code {code} outside [0, {size})"
                    )
                allowed_sets[level][normalized[:level]].add(code)
            n_codes += 1
        if n_codes == 0:
            raise ValueError("cannot build a prefix index from empty SID codes")

        self.allowed_next = tuple(
            {
                prefix: tuple(sorted(values))
                for prefix, values in level.items()
            }
            for level in allowed_sets
        )

    def allowed(self, prefix: Sequence[int]) -> tuple[int, ...]:
        level = len(prefix)
        if level >= len(self.codebook_sizes):
            return ()
        return self.allowed_next[level].get(tuple(prefix), ())

    def contains(self, codes: Sequence[int]) -> bool:
        normalized = tuple(int(code) for code in codes)
        if len(normalized) != len(self.codebook_sizes):
            return False
        return normalized[-1] in self.allowed(normalized[:-1])


def _to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


@dataclass
class SidRankingMetrics:
    """Accumulate fixed Top-10 SID generation metrics."""

    prefix_index: SidPrefixIndex
    n_samples: int = 0
    n_hits: int = 0
    reciprocal_rank_sum: float = 0.0
    level_hits: np.ndarray = field(init=False)
    level_reciprocal_rank_sums: np.ndarray = field(init=False)
    n_invalid: int = 0
    n_predictions: int = 0

    def __post_init__(self) -> None:
        self.level_hits = np.zeros(
            len(self.prefix_index.codebook_sizes),
            dtype=np.int64,
        )
        self.level_reciprocal_rank_sums = np.zeros(
            len(self.prefix_index.codebook_sizes),
            dtype=np.float64,
        )

    def update(self, predictions, targets) -> None:
        predicted = _to_numpy(predictions)
        target = _to_numpy(targets)
        n_levels = len(self.prefix_index.codebook_sizes)
        if predicted.ndim != 3 or predicted.shape[2] != n_levels:
            raise ValueError(
                f"predictions must have shape [N,K,{n_levels}]"
            )
        if predicted.shape[1] < EVALUATION_TOP_K:
            raise ValueError(
                f"predictions must contain at least {EVALUATION_TOP_K} beams"
            )
        if target.shape != (predicted.shape[0], n_levels):
            raise ValueError(
                f"targets must have shape [{predicted.shape[0]},{n_levels}]"
            )
        top_predictions = predicted[:, :EVALUATION_TOP_K].astype(
            np.int64,
            copy=False,
        )
        target = target.astype(np.int64, copy=False)
        matches = np.all(
            top_predictions == target[:, None, :],
            axis=-1,
        )
        has_hit = matches.any(axis=1)
        first_ranks = matches.argmax(axis=1) + 1

        self.n_samples += int(target.shape[0])
        self.n_hits += int(has_hit.sum())
        self.reciprocal_rank_sum += float(
            np.where(has_hit, 1.0 / first_ranks, 0.0).sum()
        )
        level_matches = (
            top_predictions == target[:, None, :]
        ).transpose(0, 2, 1)
        level_has_hit = level_matches.any(axis=2)
        level_first_ranks = level_matches.argmax(axis=2) + 1
        self.level_hits += level_has_hit.sum(axis=0)
        self.level_reciprocal_rank_sums += np.where(
            level_has_hit,
            1.0 / level_first_ranks,
            0.0,
        ).sum(axis=0)
        self.n_predictions += int(
            top_predictions.shape[0] * EVALUATION_TOP_K
        )
        self.n_invalid += sum(
            not self.prefix_index.contains(codes)
            for sample in top_predictions
            for codes in sample
        )

    def compute(self) -> dict[str, float]:
        if self.n_samples == 0:
            raise ValueError("cannot compute metrics without samples")
        metrics = {}
        for level in range(len(self.level_hits)):
            metrics[f"sid{level}_hr_at_10"] = (
                int(self.level_hits[level]) / self.n_samples
            )
            metrics[f"sid{level}_mrr_at_10"] = (
                float(self.level_reciprocal_rank_sums[level])
                / self.n_samples
            )
        metrics["overall_hr_at_10"] = self.n_hits / self.n_samples
        metrics["overall_mrr_at_10"] = (
            self.reciprocal_rank_sum / self.n_samples
        )
        metrics["invalid_sid_rate"] = (
            self.n_invalid / self.n_predictions
            if self.n_predictions
            else 0.0
        )
        return metrics
