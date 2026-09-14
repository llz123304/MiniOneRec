"""Date-ordered mini-batch sampling for chronological recommendation training."""

from __future__ import annotations

import math
import random
from array import array
from collections import defaultdict
from typing import Dict, Iterator, Sequence


class DayBatchSampler:
    """Yield date-pure batches while visiting dates in chronological order.

    Samples are shuffled within each date. The incomplete final batch of a date
    is retained without padding, so gradient accumulation may span adjacent
    dates even though an individual mini-batch never does.
    """

    def __init__(
        self,
        sample_dates: Sequence[int],
        batch_size: int,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(sample_dates) == 0:
            raise ValueError("sample_dates must not be empty")

        self.micro_batch_size = int(batch_size)
        # Accelerate uses ``batch_size=None`` to recognize variable-size batches.
        self.batch_size = None
        self.drop_last = False
        self.seed = int(seed)

        indices_by_date: Dict[int, array] = defaultdict(
            lambda: array("I")
        )
        for index, date in enumerate(sample_dates):
            indices_by_date[int(date)].append(index)
        self.indices_by_date = dict(indices_by_date)
        self.dates = sorted(self.indices_by_date)

        self.batch_counts = {
            date: math.ceil(len(indices) / self.micro_batch_size)
            for date, indices in self.indices_by_date.items()
        }

    def __len__(self) -> int:
        return sum(self.batch_counts.values())

    def __iter__(self) -> Iterator[Sequence[int]]:
        rng = random.Random(self.seed)
        for date in self.dates:
            indices = array("I", self.indices_by_date[date])
            rng.shuffle(indices)

            for start in range(0, len(indices), self.micro_batch_size):
                yield indices[start : start + self.micro_batch_size]
