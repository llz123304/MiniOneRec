"""Checks for the KuaiRand exposure-history feature contract."""

import unittest

from lazy_onerec.data.schema import (
    BINARY_EVENT_FIELDS,
    REQUEST_CATEGORICAL_CARDINALITIES,
    REQUEST_CATEGORICAL_FIELDS,
    encode_category_with_unknown,
)


class KuaiRandExposureSchemaTest(unittest.TestCase):
    def test_click_is_retained_as_historical_exposure_feedback(self):
        self.assertIn("is_click", BINARY_EVENT_FIELDS)

    def test_binary_event_fields_are_unique(self):
        self.assertEqual(len(BINARY_EVENT_FIELDS), len(set(BINARY_EVENT_FIELDS)))

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
