"""Checks for the KuaiRand exposure-history feature contract."""

import unittest

from lazy_onerec.data.schema import (
    CONTEXT_SEQUENCE_NAMES,
    DEFAULT_GID_SEQUENCE_LENGTHS,
    DEFAULT_QFORMER_QUERY_COUNTS,
    DEEP_INTERACTION_FIELDS,
    GID_SEQUENCE_NAMES,
    N_LONG_VIEW_DURATION_BUCKETS,
    N_STATIC_CONTEXT_TOKENS,
    N_USER_REQUEST_CONTEXT_TOKENS,
    REQUEST_CATEGORICAL_CARDINALITIES,
    REQUEST_CATEGORICAL_FIELDS,
    bucket_long_view_duration_ms,
    encode_category_with_unknown,
    recent_behavior_bounds,
    raw_context_length,
    total_context_length,
)


class KuaiRandExposureSchemaTest(unittest.TestCase):
    def test_behavior_histories_are_independent(self):
        self.assertEqual(
            GID_SEQUENCE_NAMES,
            ("click", "long_view", "like", "deep_interact", "hate"),
        )
        self.assertEqual(
            CONTEXT_SEQUENCE_NAMES,
            (
                "click",
                "long_view",
                "long_view_duration",
                "like",
                "deep_interact",
                "hate",
            ),
        )

    def test_default_sequence_lengths_are_asymmetric(self):
        self.assertEqual(
            DEFAULT_GID_SEQUENCE_LENGTHS,
            {
                "click": 128,
                "long_view": 128,
                "like": 64,
                "deep_interact": 32,
                "hate": 16,
            },
        )
        self.assertEqual(
            raw_context_length(DEFAULT_GID_SEQUENCE_LENGTHS),
            498,
        )
        self.assertEqual(
            DEFAULT_QFORMER_QUERY_COUNTS,
            {
                "click": 16,
                "long_view": 16,
                "long_view_duration": 16,
                "like": 8,
                "deep_interact": 4,
                "hate": 2,
            },
        )
        self.assertEqual(
            total_context_length(DEFAULT_GID_SEQUENCE_LENGTHS),
            64,
        )
        self.assertEqual(N_USER_REQUEST_CONTEXT_TOKENS, 2)
        self.assertEqual(N_STATIC_CONTEXT_TOKENS, 2)

    def test_sparse_deep_interactions_are_merged(self):
        self.assertEqual(
            DEEP_INTERACTION_FIELDS,
            ("is_follow", "is_comment", "is_forward"),
        )

    def test_long_view_duration_bucket(self):
        self.assertEqual(N_LONG_VIEW_DURATION_BUCKETS, 100)
        self.assertEqual(bucket_long_view_duration_ms(-1), 0)
        self.assertEqual(bucket_long_view_duration_ms(1_000), 1)
        self.assertEqual(bucket_long_view_duration_ms(4_000), 2)
        self.assertEqual(bucket_long_view_duration_ms(10_000_000), 99)

    def test_behavior_history_strictly_precedes_target(self):
        event_times = [100, 200, 300, 400, 400, 500]
        start, end = recent_behavior_bounds(
            event_times,
            target_time=400,
            max_history=2,
        )
        self.assertEqual(event_times[start:end], [200, 300])

    def test_behavior_histories_truncate_independently(self):
        dense_times = list(range(200))
        sparse_times = [4, 80, 160]
        dense_bounds = recent_behavior_bounds(dense_times, 180, 128)
        sparse_bounds = recent_behavior_bounds(sparse_times, 180, 128)
        self.assertEqual(
            dense_times[slice(*dense_bounds)],
            list(range(52, 180)),
        )
        self.assertEqual(
            sparse_times[slice(*sparse_bounds)],
            sparse_times,
        )

    def test_request_feature_contract(self):
        self.assertEqual(
            REQUEST_CATEGORICAL_FIELDS,
            (
                "request_tab",
                "request_hour",
                "request_day_of_week",
                "request_time_gap_bucket",
            ),
        )
        self.assertEqual(
            REQUEST_CATEGORICAL_CARDINALITIES,
            (16, 25, 8, 9),
        )

    def test_request_categories_reserve_zero_for_unknown(self):
        self.assertEqual(encode_category_with_unknown(0, 15), 1)
        self.assertEqual(encode_category_with_unknown(14, 15), 15)
        self.assertEqual(encode_category_with_unknown(-1, 15), 0)
        self.assertEqual(encode_category_with_unknown(15, 15), 0)


if __name__ == "__main__":
    unittest.main()
