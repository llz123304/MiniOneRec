"""Tests for chronological, date-pure training batches."""

import unittest

from lazy_onerec.data.day_batch_sampler import DayBatchSampler


class DayBatchSamplerTest(unittest.TestCase):
    def test_dates_are_ordered_and_batches_do_not_mix_dates(self):
        sample_dates = [20220409] * 5 + [20220408] * 7
        sampler = DayBatchSampler(
            sample_dates,
            batch_size=2,
        )

        batches = list(sampler)
        batch_dates = [
            {sample_dates[index] for index in batch}
            for batch in batches
        ]

        self.assertTrue(all(len(dates) == 1 for dates in batch_dates))
        ordered_dates = [next(iter(dates)) for dates in batch_dates]
        self.assertEqual(ordered_dates, sorted(ordered_dates))
        self.assertTrue(all(1 <= len(batch) <= 2 for batch in batches))
        sampled_indices = [index for batch in batches for index in batch]
        self.assertCountEqual(sampled_indices, range(len(sample_dates)))

    def test_partial_batch_is_retained_for_each_date(self):
        sample_dates = [1] * 7 + [2] * 5
        sampler = DayBatchSampler(
            sample_dates,
            batch_size=2,
        )
        batches = list(sampler)

        self.assertEqual(sampler.batch_counts, {1: 4, 2: 3})
        self.assertEqual([len(batch) for batch in batches], [2, 2, 2, 1, 2, 2, 1])
        self.assertEqual(sum(map(len, batches)), len(sample_dates))

    def test_shuffle_is_reproducible(self):
        sample_dates = [1] * 32
        sampler = DayBatchSampler(
            sample_dates,
            batch_size=4,
            seed=7,
        )
        self.assertEqual(list(sampler), list(sampler))

    def test_date_smaller_than_batch_size_is_retained(self):
        sampler = DayBatchSampler([20220408] * 3, batch_size=8)

        self.assertEqual([len(batch) for batch in sampler], [3])


if __name__ == "__main__":
    unittest.main()
