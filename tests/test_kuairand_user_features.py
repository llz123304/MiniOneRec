"""Tests for static KuaiRand user-feature preprocessing."""

import unittest
from pathlib import Path

import numpy as np

from lazy_onerec.data.user_features import (
    CATEGORICAL_USER_FIELDS,
    CONTINUOUS_USER_FIELDS,
    KuaiRandUserFeatureStore,
)


class KuaiRandUserFeatureStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).parents[1]
            / "lazy_onerec/KuaiRand-1K/data/user_features_1k.csv"
        )
        cls.store = KuaiRandUserFeatureStore.load(path)

    def test_feature_shapes_and_unknown_row(self):
        self.assertEqual(self.store.categorical.shape, (1001, 26))
        self.assertEqual(self.store.continuous.shape, (1001, 4))
        self.assertEqual(len(self.store.categorical_cardinalities), 26)
        np.testing.assert_array_equal(self.store.categorical[0], 0)
        np.testing.assert_array_equal(self.store.continuous[0], 0.0)

    def test_user_12_matches_source_categories(self):
        row = self.store.row_for_user(12)
        self.assertNotEqual(row, 0)
        live_index = CATEGORICAL_USER_FIELDS.index("is_live_streamer")
        onehot3_index = CATEGORICAL_USER_FIELDS.index("onehot_feat3")

        self.assertEqual(self.store.categorical[row, 0], 13)
        self.assertEqual(self.store.categorical[row, live_index], 0)
        self.assertEqual(self.store.categorical[row, onehot3_index], 649)

    def test_continuous_features_are_finite_and_standardized(self):
        values = self.store.continuous[1:]
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(values.std(axis=0), 1.0, atol=1e-5)

    def test_feature_contract_excludes_constant_column(self):
        all_fields = set(CATEGORICAL_USER_FIELDS) | set(
            CONTINUOUS_USER_FIELDS
        )
        self.assertNotIn("is_lowactive_period", all_fields)
        self.assertEqual(self.store.row_for_user(10000), 0)


if __name__ == "__main__":
    unittest.main()
