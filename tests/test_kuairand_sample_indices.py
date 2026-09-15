"""Regression tests for KuaiRand temporal sample indexing."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lazy_onerec.data.kuairand import (
    _local_calendar_from_time_ms,
    build_exposure_sample_indices,
)


class KuaiRandSampleIndexTest(unittest.TestCase):
    def test_calendar_is_derived_in_shanghai_timezone(self):
        local_time = pd.DatetimeIndex(
            [
                "2022-04-08 00:30:00+08:00",
                "2022-04-10 23:59:00+08:00",
            ]
        )
        time_ms = pd.Series(local_time.astype("int64") // 1_000_000)

        dates, weekdays = _local_calendar_from_time_ms(time_ms)

        self.assertEqual(dates.tolist(), [20220408, 20220410])
        self.assertEqual(weekdays.tolist(), [4, 6])

    def test_min_history_counts_only_strictly_earlier_rows(self):
        sequence = SimpleNamespace(
            gid_ids=np.arange(1, 8, dtype=np.int32),
            time_ms=np.asarray([100, 100, 200, 200, 200, 300, 400]),
            dates=np.asarray(
                [
                    20220408,
                    20220408,
                    20220409,
                    20220409,
                    20220409,
                    20220409,
                    20220410,
                ],
                dtype=np.int32,
            ),
        )
        corpus = SimpleNamespace(sequences=[sequence])
        artifact = SimpleNamespace(
            item_codes={str(video_id): (0, 0, 0) for video_id in range(7)}
        )

        indices = build_exposure_sample_indices(
            corpus=corpus,
            sid_artifact=artifact,
            min_history=3,
            warmup_days=1,
            test_days=1,
        )

        self.assertEqual(indices["train"].target_positions.tolist(), [5])
        self.assertEqual(indices["test"].target_positions.tolist(), [6])


if __name__ == "__main__":
    unittest.main()
