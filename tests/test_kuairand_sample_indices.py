"""Regression tests for KuaiRand temporal sample indexing."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lazy_onerec.data.kuairand import (
    _local_calendar_from_time_ms,
    _positive_target_mask,
    build_exposure_sample_indices,
)


class KuaiRandSampleIndexTest(unittest.TestCase):
    def test_positive_target_rule_and_hate_precedence(self):
        frame = pd.DataFrame(
            {
                "is_click": [0, 1, 1, 0],
                "long_view": [0, 0, 0, 0],
                "is_like": [0, 0, 0, 0],
                "is_follow": [0, 0, 0, 0],
                "is_comment": [0, 0, 0, 0],
                "is_forward": [0, 0, 0, 0],
                "is_profile_enter": [1, 0, 0, 0],
                "is_hate": [0, 0, 1, 0],
            }
        )

        self.assertEqual(
            _positive_target_mask(frame).tolist(),
            [True, True, False, False],
        )

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
            positive_target=np.asarray(
                [False, False, False, True, False, True, True],
                dtype=np.bool_,
            ),
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
        self.assertEqual(len(indices["train"]), 1)
        self.assertEqual(len(indices["test"]), 1)

    def test_only_positive_non_hate_targets_are_indexed(self):
        sequence = SimpleNamespace(
            gid_ids=np.arange(1, 6, dtype=np.int32),
            time_ms=np.asarray([100, 200, 300, 400, 500]),
            positive_target=np.asarray(
                [False, False, False, False, True],
                dtype=np.bool_,
            ),
            dates=np.asarray(
                [20220408, 20220408, 20220408, 20220409, 20220410],
                dtype=np.int32,
            ),
        )
        corpus = SimpleNamespace(sequences=[sequence])
        artifact = SimpleNamespace(
            item_codes={str(video_id): (0, 0, 0) for video_id in range(5)}
        )

        indices = build_exposure_sample_indices(
            corpus=corpus,
            sid_artifact=artifact,
            min_history=3,
            warmup_days=1,
            test_days=1,
        )

        self.assertEqual(indices["train"].target_positions.tolist(), [])
        self.assertEqual(indices["test"].target_positions.tolist(), [4])


if __name__ == "__main__":
    unittest.main()
