"""Validate pipeline artifacts and record the configuration that produced them."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGES = ("embedding", "sid", "train", "evaluate")
MANIFEST_NAME = "pipeline_stage.json"
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


@dataclass(frozen=True)
class PipelineSettings:
    data_root: str
    embedding_model: str
    embedding_revision: str
    text_dir: str
    embedding_dir: str
    sid_method: str
    sid_codebook_sizes: tuple[int, ...]
    sid_distance_metric: str
    sid_dir: str
    positive_target: str
    num_train_epochs: int
    model_dir: str
    evaluation_dir: str


def slugify(value: str) -> str:
    """Return a stable path component for a model name or revision."""
    return re.sub(r"[^a-z0-9]+", "-", value.rstrip("/").lower()).strip("-")


def _nonempty(directory: Path, name: str) -> bool:
    path = directory / name
    return path.is_file() and path.stat().st_size > 0


def artifacts_complete(stage: str, directory: Path) -> bool:
    """Return whether a stage directory contains its complete output contract."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    try:
        if stage == "embedding":
            config = json.loads(
                (directory / "embedding_config.json").read_text(
                    encoding="utf-8"
                )
            )
            progress = json.loads(
                (directory / "progress.json").read_text(encoding="utf-8")
            )
            return (
                config.get("completed") is True
                and progress.get("next_index") == config.get("num_items")
                and progress.get("num_items") == config.get("num_items")
                and _nonempty(directory, "item_embeddings.npy")
                and _nonempty(directory, "item_ids.npy")
            )
        if stage == "sid":
            return all(
                _nonempty(directory, name)
                for name in (
                    "sid_index.json",
                    "codes.npy",
                    "codebooks.npz",
                    "sid_metrics.json",
                )
            )
        if stage == "train":
            return _nonempty(directory, "config.json") and any(
                _nonempty(directory, name) for name in MODEL_WEIGHT_FILES
            )

        metrics = json.loads(
            (directory / "test_sid_metrics.json").read_text(encoding="utf-8")
        )
        required = {"overall_hr_at_10", "overall_mrr_at_10", "invalid_sid_rate"}
        for level in range(3):
            required.update(
                {f"sid{level}_hr_at_10", f"sid{level}_mrr_at_10"}
            )
        return required.issubset(metrics)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return False


def expected_manifest(
    stage: str,
    settings: PipelineSettings,
) -> dict[str, Any]:
    """Build the stage-specific configuration fingerprint stored on success."""
    if stage == "embedding":
        inputs = {
            "data_root": settings.data_root,
            "model": settings.embedding_model,
            "revision": settings.embedding_revision or None,
            "text_dir": settings.text_dir,
        }
        output_dir = settings.embedding_dir
    elif stage == "sid":
        inputs = {
            "embedding_dir": settings.embedding_dir,
            "method": settings.sid_method,
            "codebook_sizes": list(settings.sid_codebook_sizes),
            "distance_metric": settings.sid_distance_metric,
        }
        output_dir = settings.sid_dir
    elif stage == "train":
        inputs = {
            "data_root": settings.data_root,
            "sid_dir": settings.sid_dir,
            "positive_target": settings.positive_target,
            "num_train_epochs": settings.num_train_epochs,
        }
        output_dir = settings.model_dir
    elif stage == "evaluate":
        inputs = {
            "data_root": settings.data_root,
            "sid_dir": settings.sid_dir,
            "model_dir": settings.model_dir,
            "positive_target": settings.positive_target,
            "num_train_epochs": settings.num_train_epochs,
        }
        output_dir = settings.evaluation_dir
    else:
        raise ValueError(f"unknown stage: {stage}")

    return {
        "format": "lazy-onerec-pipeline-stage",
        "version": 2,
        "stage": stage,
        "inputs": inputs,
        "output_dir": output_dir,
    }


def stage_complete(
    stage: str,
    directory: Path,
    settings: PipelineSettings,
) -> bool:
    """Require both valid artifacts and their matching pipeline manifest."""
    if not artifacts_complete(stage, directory):
        return False
    try:
        actual = json.loads(
            (directory / MANIFEST_NAME).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return actual == expected_manifest(stage, settings)


def mark_stage(
    stage: str,
    directory: Path,
    settings: PipelineSettings,
) -> None:
    """Atomically mark a stage only after its artifact contract is satisfied."""
    if not artifacts_complete(stage, directory):
        raise ValueError(
            f"{stage} finished without complete artifacts: {directory}"
        )
    path = directory / MANIFEST_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            expected_manifest(stage, settings),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", default="")
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--sid-method", required=True)
    parser.add_argument(
        "--sid-codebook-sizes",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument("--sid-distance-metric", required=True)
    parser.add_argument("--sid-dir", required=True)
    parser.add_argument("--positive-target", required=True)
    parser.add_argument("--num-train-epochs", type=int, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)


def _settings_from_args(args: argparse.Namespace) -> PipelineSettings:
    return PipelineSettings(
        data_root=args.data_root,
        embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision,
        text_dir=args.text_dir,
        embedding_dir=args.embedding_dir,
        sid_method=args.sid_method,
        sid_codebook_sizes=tuple(args.sid_codebook_sizes),
        sid_distance_metric=args.sid_distance_metric,
        sid_dir=args.sid_dir,
        positive_target=args.positive_target,
        num_train_epochs=args.num_train_epochs,
        model_dir=args.model_dir,
        evaluation_dir=args.evaluation_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    slug_parser = commands.add_parser("slug")
    slug_parser.add_argument("value")
    for name in ("check", "mark"):
        stage_parser = commands.add_parser(name)
        stage_parser.add_argument("--stage", choices=STAGES, required=True)
        stage_parser.add_argument("--directory", type=Path, required=True)
        _add_settings_arguments(stage_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "slug":
        print(slugify(args.value))
        return

    settings = _settings_from_args(args)
    if args.command == "check":
        raise SystemExit(
            0 if stage_complete(args.stage, args.directory, settings) else 1
        )
    try:
        mark_stage(args.stage, args.directory, settings)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
