"""Pipeline completion and manifest regression tests."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from lazy_onerec.src.pipeline_artifacts import (
    MANIFEST_NAME,
    PipelineSettings,
    mark_stage,
    stage_complete,
)


def _settings(root: Path) -> PipelineSettings:
    return PipelineSettings(
        data_root="data",
        embedding_model="example/model",
        embedding_revision="revision",
        text_dir=str(root / "texts"),
        embedding_dir=str(root / "embedding"),
        sid_method="rq-kmeans",
        sid_codebook_sizes=(8, 8, 8),
        sid_distance_metric="cosine",
        sid_dir=str(root / "sid"),
        positive_target="all",
        num_train_epochs=1,
        model_dir=str(root / "model"),
        evaluation_dir=str(root / "evaluation"),
    )


def _write_nonempty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


class PipelineArtifactsTest(unittest.TestCase):
    def test_configuration_change_invalidates_completed_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)
            model_dir = Path(settings.model_dir)
            _write_nonempty(model_dir / "config.json")
            _write_nonempty(model_dir / "model.safetensors")

            mark_stage("train", model_dir, settings)

            self.assertTrue(stage_complete("train", model_dir, settings))
            changed = replace(settings, num_train_epochs=2)
            self.assertFalse(stage_complete("train", model_dir, changed))

    def test_partial_artifacts_are_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)
            model_dir = Path(settings.model_dir)
            _write_nonempty(model_dir / "config.json")

            self.assertFalse(stage_complete("train", model_dir, settings))

    def test_failed_stage_cannot_write_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)
            model_dir = Path(settings.model_dir)
            model_dir.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "without complete artifacts"):
                mark_stage("train", model_dir, settings)

            self.assertFalse((model_dir / MANIFEST_NAME).exists())

    def test_downstream_change_does_not_invalidate_embedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)
            embedding_dir = Path(settings.embedding_dir)
            _write_nonempty(embedding_dir / "item_embeddings.npy")
            _write_nonempty(embedding_dir / "item_ids.npy")
            (embedding_dir / "embedding_config.json").write_text(
                json.dumps({"completed": True, "num_items": 4}),
                encoding="utf-8",
            )
            (embedding_dir / "progress.json").write_text(
                json.dumps({"next_index": 4, "num_items": 4}),
                encoding="utf-8",
            )
            mark_stage("embedding", embedding_dir, settings)

            changed = replace(settings, num_train_epochs=3)

            self.assertTrue(stage_complete("embedding", embedding_dir, changed))


if __name__ == "__main__":
    unittest.main()
