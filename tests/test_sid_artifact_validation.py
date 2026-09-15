"""Validation tests for semantic-ID row construction."""

import unittest

from lazy_onerec.sid.artifact import SemanticIDArtifact


class SemanticIDArtifactValidationTest(unittest.TestCase):
    def test_duplicate_normalized_item_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate item_id"):
            SemanticIDArtifact.from_rows(
                item_ids=[1, "1"],
                codes=[(0, 0, 0), (1, 1, 1)],
                codebook_sizes=(2, 2, 2),
            )

    def test_more_item_ids_than_codes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            SemanticIDArtifact.from_rows(
                item_ids=["a", "b"],
                codes=[(0, 0, 0)],
                codebook_sizes=(2, 2, 2),
            )

    def test_more_codes_than_item_ids_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            SemanticIDArtifact.from_rows(
                item_ids=["a"],
                codes=[(0, 0, 0), (1, 1, 1)],
                codebook_sizes=(2, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()
