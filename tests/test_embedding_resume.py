"""Embedding output initialization and resume regressions."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lazy_onerec.src.embed_kuairand_items import (
    EmbeddingOutputPaths,
    persist_embedding_batch,
    prepare_encoding_output,
)


def _paths(directory: Path) -> EmbeddingOutputPaths:
    return EmbeddingOutputPaths(
        directory=directory,
        embeddings=directory / "item_embeddings.npy",
        item_ids=directory / "item_ids.npy",
        config=directory / "embedding_config.json",
        progress=directory / "progress.json",
    )


class EmbeddingResumeTest(unittest.TestCase):
    def test_new_output_resumes_from_persisted_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary) / "embeddings")
            args = SimpleNamespace(overwrite=False, storage_dtype="float32")
            item_ids = np.array([3, 5, 8], dtype=np.int32)
            config = {"num_items": 3, "output_dim": 2}
            state = prepare_encoding_output(
                args,
                paths,
                item_ids,
                output_dim=2,
                config=config,
            )
            persist_embedding_batch(
                state.embeddings,
                paths.progress,
                batch_start=0,
                values=np.ones((2, 2), dtype=np.float32),
                storage_dtype="float32",
            )

            resumed = prepare_encoding_output(
                args,
                paths,
                item_ids,
                output_dim=2,
                config=config,
            )

            self.assertEqual(resumed.start, 2)
            self.assertTrue(np.all(resumed.embeddings[:2] == 1.0))

    def test_partial_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary) / "embeddings")
            paths.directory.mkdir()
            paths.config.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(overwrite=False, storage_dtype="float32")

            with self.assertRaisesRegex(ValueError, "incomplete output"):
                prepare_encoding_output(
                    args,
                    paths,
                    np.array([1], dtype=np.int32),
                    output_dim=2,
                    config={"num_items": 1},
                )

    def test_batch_flush_precedes_progress_update(self):
        events = []

        class FakeMemmap:
            def __len__(self):
                return 4

            def __setitem__(self, _key, _value):
                events.append("write")

            def flush(self):
                events.append("flush")

        with patch(
            "lazy_onerec.src.embed_kuairand_items.atomic_json",
            side_effect=lambda *_args: events.append("progress"),
        ):
            end = persist_embedding_batch(
                FakeMemmap(),
                Path("progress.json"),
                batch_start=1,
                values=np.ones((2, 3), dtype=np.float32),
                storage_dtype="float32",
            )

        self.assertEqual(end, 3)
        self.assertEqual(events, ["write", "flush", "progress"])


if __name__ == "__main__":
    unittest.main()
