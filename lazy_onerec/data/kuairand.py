"""KuaiRand exposure-sequence adapter for next-SID generation.

History uses an independent GID sequence slot. Target GID is only used to look
up the semantic-ID label and is never inserted into its own history.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import (
    BINARY_EVENT_FIELDS,
    N_TAB_EMBEDDINGS,
    N_TIME_GAP_BUCKETS,
    encode_category_with_unknown,
)
from .user_features import KuaiRandUserFeatureStore
from ..sid.artifact import SemanticIDArtifact
from ..sid.token_codec import SidTokenCodec


LOG_COLUMNS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    *BINARY_EVENT_FIELDS,
    "play_time_ms",
    "duration_ms",
    "tab",
)

LOG_DTYPES = {
    "user_id": np.int32,
    "video_id": np.int32,
    "date": np.int32,
    "hourmin": np.int16,
    "time_ms": np.int64,
    **{field: np.int8 for field in BINARY_EVENT_FIELDS},
    "play_time_ms": np.int32,
    "duration_ms": np.int32,
    "tab": np.int8,
}

PLAY_RATIO_BINS = np.asarray([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
TIME_GAP_BINS_SECONDS = np.asarray(
    [60, 300, 1_800, 21_600, 86_400, 604_800, 2_592_000]
)


@dataclass
class UserExposureSequence:
    user_feature_row: int
    gid_ids: np.ndarray
    dates: np.ndarray
    hour: np.ndarray
    day_of_week: np.ndarray
    binary: Dict[str, np.ndarray]
    play_ratio_bucket: np.ndarray
    time_gap_bucket: np.ndarray
    tab: np.ndarray


@dataclass
class ExposureSampleIndex:
    sequence_indices: array
    target_positions: array
    dates: array

    @classmethod
    def empty(cls) -> "ExposureSampleIndex":
        return cls(array("I"), array("I"), array("i"))


def _read_standard_logs(data_dir: Path) -> pd.DataFrame:
    def find_one(pattern: str) -> Path:
        matches = sorted(data_dir.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one {pattern!r} under {data_dir}, found {matches}"
            )
        return matches[0]

    paths = [
        find_one("log_standard_4_08_to_4_21_*.csv"),
        find_one("log_standard_4_22_to_5_08_*.csv"),
    ]

    frames = [
        pd.read_csv(
            path,
            usecols=list(LOG_COLUMNS),
            dtype=LOG_DTYPES,
        )
        for path in paths
    ]
    logs = pd.concat(frames, ignore_index=True)
    logs.sort_values(["user_id", "time_ms"], inplace=True)

    duration = logs["duration_ms"].clip(lower=1)
    ratio = (logs["play_time_ms"] / duration).clip(lower=0, upper=10)
    logs["play_ratio_bucket"] = np.digitize(ratio, PLAY_RATIO_BINS)

    gap_seconds = (
        logs.groupby("user_id", sort=False)["time_ms"]
        .diff()
        .fillna(0)
        .clip(lower=0)
        / 1000.0
    )
    logs["time_gap_bucket"] = np.digitize(
        gap_seconds, TIME_GAP_BINS_SECONDS
    )
    hour = logs["hourmin"] // 100
    logs["request_hour"] = hour.where(hour.between(0, 23), -1).astype(np.int8)
    date_to_weekday = {
        int(date): datetime.strptime(str(int(date)), "%Y%m%d").weekday()
        for date in logs["date"].unique()
    }
    logs["request_day_of_week"] = (
        logs["date"].map(date_to_weekday).astype(np.int8)
    )
    return logs


class KuaiRandExposureCorpus:
    """In-memory standard-exposure sequences shared by dataset splits."""

    def __init__(
        self,
        data_root: str | Path,
    ):
        data_root = Path(data_root)
        data_dir = data_root / "data" if (data_root / "data").is_dir() else data_root
        logs = _read_standard_logs(data_dir)
        user_feature_matches = sorted(data_dir.glob("user_features_*.csv"))
        if len(user_feature_matches) != 1:
            raise FileNotFoundError(
                f"expected one user_features_*.csv under {data_dir}, "
                f"found {user_feature_matches}"
            )
        self.user_features = KuaiRandUserFeatureStore.load(
            user_feature_matches[0]
        )
        item_feature_matches = sorted(
            data_dir.glob("video_features_basic_*.csv")
        )
        if len(item_feature_matches) == 1:
            item_features = item_feature_matches[0]
            catalog_ids = pd.read_csv(item_features, usecols=["video_id"])
            max_video_id = int(catalog_ids["video_id"].max())
        else:
            max_video_id = int(logs["video_id"].max())
        # Actual GID is video_id + 1; row 0 remains padding.
        self.num_gid_embeddings = max_video_id + 2
        self.sequences: List[UserExposureSequence] = []
        for user_id, group in logs.groupby("user_id", sort=False):
            group = group.reset_index(drop=True)
            user_feature_row = self.user_features.row_for_user(int(user_id))
            if user_feature_row == 0:
                raise ValueError(
                    f"user_id={int(user_id)} is missing from user_features"
                )
            self.sequences.append(
                UserExposureSequence(
                    user_feature_row=user_feature_row,
                    gid_ids=group["video_id"].to_numpy(np.int32) + 1,
                    dates=group["date"].to_numpy(np.int32),
                    hour=group["request_hour"].to_numpy(np.int8),
                    day_of_week=group[
                        "request_day_of_week"
                    ].to_numpy(np.int8),
                    binary={
                        field: group[field].to_numpy(np.int8)
                        for field in BINARY_EVENT_FIELDS
                    },
                    play_ratio_bucket=group[
                        "play_ratio_bucket"
                    ].to_numpy(np.int8),
                    time_gap_bucket=group[
                        "time_gap_bucket"
                    ].to_numpy(np.int8),
                    tab=group["tab"].fillna(0).to_numpy(np.int8),
                )
            )


def build_exposure_sample_indices(
    corpus: KuaiRandExposureCorpus,
    sid_artifact: SemanticIDArtifact,
    min_history: int,
    train_end_date: int = 20220421,
    valid_end_date: int = 20220429,
) -> Dict[str, ExposureSampleIndex]:
    indices = {
        split: ExposureSampleIndex.empty()
        for split in ("train", "valid", "test")
    }
    for sequence_index, sequence in enumerate(corpus.sequences):
        for position in range(min_history, len(sequence.gid_ids)):
            target_video_id = str(int(sequence.gid_ids[position]) - 1)
            if target_video_id not in sid_artifact.item_codes:
                continue
            date = int(sequence.dates[position])
            if date <= train_end_date:
                split = "train"
            elif date <= valid_end_date:
                split = "valid"
            else:
                split = "test"
            sample_index = indices[split]
            sample_index.sequence_indices.append(sequence_index)
            sample_index.target_positions.append(position)
            sample_index.dates.append(date)
    return indices


class KuaiRandNextExposureSidDataset(Dataset):
    """Rolling next-exposure SID samples from a shared KuaiRand corpus."""

    def __init__(
        self,
        corpus: KuaiRandExposureCorpus,
        sid_artifact: SemanticIDArtifact,
        sample_index: ExposureSampleIndex,
        max_history: int = 128,
        sample: int = -1,
        seed: int = 42,
    ):
        self.sid_artifact = sid_artifact
        self.sid_codec = SidTokenCodec(sid_artifact.codebook_sizes)
        self.max_history = max_history
        self.sequences = corpus.sequences
        self.user_features = corpus.user_features
        self.sample_sequence_indices = sample_index.sequence_indices
        self.sample_target_positions = sample_index.target_positions
        self.sample_dates = sample_index.dates

        if sample > 0 and sample < len(self):
            rng = np.random.RandomState(seed)
            chosen = rng.choice(len(self), size=sample, replace=False)
            chosen_indices = chosen.tolist()
            self.sample_sequence_indices = array(
                "I",
                (
                    self.sample_sequence_indices[index]
                    for index in chosen_indices
                ),
            )
            self.sample_target_positions = array(
                "I",
                (
                    self.sample_target_positions[index]
                    for index in chosen_indices
                ),
            )
            self.sample_dates = array(
                "i", (self.sample_dates[index] for index in chosen_indices)
            )

    def __len__(self) -> int:
        return len(self.sample_target_positions)

    def __getitem__(
        self, index: int
    ) -> Dict[str, Sequence[int] | Sequence[float]]:
        sequence_index = self.sample_sequence_indices[index]
        target_position = self.sample_target_positions[index]
        sequence = self.sequences[sequence_index]
        start = max(0, target_position - self.max_history)
        history_slice = slice(start, target_position)

        target_gid = int(sequence.gid_ids[target_position])
        target_video_id = str(target_gid - 1)
        target_codes = self.sid_artifact.item_codes[target_video_id]
        user_row = sequence.user_feature_row

        example: Dict[str, Sequence[int] | Sequence[float]] = {
            "user_categorical_features": self.user_features.categorical[
                user_row
            ].tolist(),
            "user_continuous_features": self.user_features.continuous[
                user_row
            ].tolist(),
            "request_categorical_features": [
                encode_category_with_unknown(
                    int(sequence.tab[target_position]),
                    N_TAB_EMBEDDINGS,
                ),
                encode_category_with_unknown(
                    int(sequence.hour[target_position]),
                    24,
                ),
                encode_category_with_unknown(
                    int(sequence.day_of_week[target_position]),
                    7,
                ),
                encode_category_with_unknown(
                    int(sequence.time_gap_bucket[target_position]),
                    N_TIME_GAP_BUCKETS,
                ),
            ],
            "context_gid_ids": sequence.gid_ids[history_slice].tolist(),
            "context_play_ratio_bucket": sequence.play_ratio_bucket[
                history_slice
            ].tolist(),
            "context_time_gap_bucket": sequence.time_gap_bucket[
                history_slice
            ].tolist(),
            "context_tab": sequence.tab[history_slice].tolist(),
            "target_input_ids": self.sid_codec.decoder_inputs(target_codes),
            "labels": self.sid_codec.decoder_labels(target_codes),
        }
        for field in BINARY_EVENT_FIELDS:
            example[f"context_{field}"] = sequence.binary[field][
                history_slice
            ].tolist()
        return example


class KuaiRandCollator:
    """Right-pad all aligned history slots and stack fixed target SID fields."""

    def __call__(
        self,
        features: List[Dict[str, Sequence[int] | Sequence[float]]],
    ) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["context_gid_ids"]) for feature in features)
        history_fields = [
            "context_gid_ids",
            "context_play_ratio_bucket",
            "context_time_gap_bucket",
            "context_tab",
            *(f"context_{field}" for field in BINARY_EVENT_FIELDS),
        ]

        batch: Dict[str, torch.Tensor] = {}
        batch["user_categorical_features"] = torch.tensor(
            [feature["user_categorical_features"] for feature in features],
            dtype=torch.long,
        )
        batch["user_continuous_features"] = torch.tensor(
            [feature["user_continuous_features"] for feature in features],
            dtype=torch.float32,
        )
        batch["request_categorical_features"] = torch.tensor(
            [feature["request_categorical_features"] for feature in features],
            dtype=torch.long,
        )
        for field in history_fields:
            padded = []
            for feature in features:
                values = list(feature[field])
                padded.append(values + [0] * (max_length - len(values)))
            batch[field] = torch.tensor(padded, dtype=torch.long)

        batch["context_attention_mask"] = torch.tensor(
            [
                [1] * len(feature["context_gid_ids"])
                + [0] * (max_length - len(feature["context_gid_ids"]))
                for feature in features
            ],
            dtype=torch.long,
        )
        batch["target_input_ids"] = torch.tensor(
            [feature["target_input_ids"] for feature in features],
            dtype=torch.long,
        )
        batch["labels"] = torch.tensor(
            [feature["labels"] for feature in features],
            dtype=torch.long,
        )
        return batch
