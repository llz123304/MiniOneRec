"""Tests for natural-day train/test splitting with warmup history days."""

import unittest

from lazy_onerec.data.schema import (
    assign_day_split,
    day_split_boundaries,
)


class DaySplitTest(unittest.TestCase):
    def setUp(self):
        # Nine distinct dates: 3 warmup, 3 train, 3 test.
        self.dates = [
            20220408,
            20220409,
            20220410,
            20220411,
            20220412,
            20220413,
            20220414,
            20220415,
            20220416,
        ]

    def test_boundaries_take_first_and_last_days(self):
        warmup, test = day_split_boundaries(
            self.dates, warmup_days=3, test_days=3
        )
        self.assertEqual(warmup, {20220408, 20220409, 20220410})
        self.assertEqual(test, {20220414, 20220415, 20220416})

    def test_warmup_days_yield_no_split(self):
        warmup, test = day_split_boundaries(
            self.dates, warmup_days=3, test_days=3
        )
        for date in (20220408, 20220409, 20220410):
            self.assertIsNone(assign_day_split(date, warmup, test))

    def test_middle_days_train_and_last_days_test(self):
        warmup, test = day_split_boundaries(
            self.dates, warmup_days=3, test_days=3
        )
        for date in (20220411, 20220412, 20220413):
            self.assertEqual(assign_day_split(date, warmup, test), "train")
        for date in (20220414, 20220415, 20220416):
            self.assertEqual(assign_day_split(date, warmup, test), "test")

    def test_duplicate_dates_collapse_before_slicing(self):
        warmup, test = day_split_boundaries(
            [d for d in self.dates for _ in range(100)],
            warmup_days=3,
            test_days=3,
        )
        self.assertEqual(warmup, {20220408, 20220409, 20220410})
        self.assertEqual(test, {20220414, 20220415, 20220416})

    def test_too_few_dates_is_rejected(self):
        with self.assertRaises(ValueError):
            day_split_boundaries(
                [20220408, 20220409, 20220410, 20220411, 20220412, 20220413],
                warmup_days=3,
                test_days=3,
            )


if __name__ == "__main__":
    unittest.main()
