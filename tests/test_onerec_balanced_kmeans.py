"""Tests for OneRec's greedy quota-balanced K-means."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lazy_onerec.sid.builders import build_onerec_balanced_kmeans
from lazy_onerec.sid.onerec_balanced_kmeans import (
    balanced_kmeans,
    residual_kmeans,
)


def _values() -> np.ndarray:
    rng = np.random.RandomState(7)
    return np.concatenate(
        (
            rng.normal(loc=-3.0, scale=0.2, size=(4, 4)),
            rng.normal(loc=0.0, scale=0.2, size=(4, 4)),
            rng.normal(loc=3.0, scale=0.2, size=(3, 4)),
        ),
        axis=0,
    ).astype(np.float32)


class OneRecBalancedKMeansTest(unittest.TestCase):
    def test_matrix_assignment_is_balanced_and_deterministic(self):
        values = _values()
        first_labels, first_centers, first_stats = balanced_kmeans(
            values,
            3,
            max_iter=10,
            seed=11,
            device="cpu",
            distance_mode="matrix",
            distance_batch_size=4,
        )
        second_labels, second_centers, _ = balanced_kmeans(
            values,
            3,
            max_iter=10,
            seed=11,
            device="cpu",
            distance_mode="matrix",
            distance_batch_size=4,
        )

        self.assertEqual(sorted(np.bincount(first_labels)), [3, 4, 4])
        self.assertTrue(np.array_equal(first_labels, second_labels))
        self.assertTrue(np.allclose(first_centers, second_centers))
        self.assertEqual(first_stats["distance_mode"], "matrix")

    def test_streaming_assignment_obeys_quotas(self):
        labels, centers, stats = balanced_kmeans(
            _values(),
            3,
            max_iter=10,
            seed=13,
            distance_metric="cosine",
            device="cpu",
            distance_mode="streaming",
            distance_batch_size=3,
        )
        matrix_labels, matrix_centers, _ = balanced_kmeans(
            _values(),
            3,
            max_iter=10,
            seed=13,
            distance_metric="cosine",
            device="cpu",
            distance_mode="matrix",
            distance_batch_size=3,
        )

        self.assertEqual(sorted(np.bincount(labels)), [3, 4, 4])
        self.assertEqual(centers.shape, (3, 4))
        self.assertTrue(np.isfinite(centers).all())
        self.assertEqual(stats["distance_mode"], "streaming")
        self.assertTrue(np.array_equal(labels, matrix_labels))
        self.assertTrue(np.allclose(centers, matrix_centers))

    def test_residual_levels_and_builder_outputs(self):
        values = np.concatenate((_values(), _values()[:1]), axis=0)
        codes, codebooks, mse, level_stats = residual_kmeans(
            values,
            (3, 2),
            max_iter=5,
            seed=17,
            device="cpu",
            distance_mode="matrix",
            distance_batch_size=4,
        )

        self.assertEqual(codes.shape, (12, 2))
        self.assertEqual([book.shape for book in codebooks], [(3, 4), (2, 4)])
        self.assertEqual(sorted(np.bincount(codes[:, 0])), [4, 4, 4])
        self.assertEqual(sorted(np.bincount(codes[:, 1])), [6, 6])
        self.assertTrue(np.isfinite(mse))
        self.assertEqual(len(level_stats), 2)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embeddings_path = root / "embeddings.npy"
            item_ids_path = root / "item_ids.npy"
            output_dir = root / "sid"
            np.save(embeddings_path, values)
            np.save(item_ids_path, np.arange(len(values)))
            artifact = build_onerec_balanced_kmeans(
                embeddings_path=str(embeddings_path),
                item_ids_path=str(item_ids_path),
                output_dir=str(output_dir),
                codebook_sizes=(3, 2),
                max_iter=5,
                seed=17,
                device="cpu",
                distance_mode="matrix",
                distance_batch_size=4,
            )

            self.assertEqual(
                artifact.metadata["method"],
                "OneRec-Balanced-Kmeans",
            )
            self.assertTrue((output_dir / "codes.npy").is_file())
            self.assertTrue((output_dir / "codebooks.npz").is_file())
            self.assertTrue((output_dir / "sid_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
