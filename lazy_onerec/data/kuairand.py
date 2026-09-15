"""KuaiRand behavior-sequence adapter for next-SID generation.

Each behavior has an independent chronological GID history. Target GID is only
used to look up the semantic-ID label and is never inserted into history.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import (
    DEFAULT_GID_SEQUENCE_LENGTHS,
    DEEP_INTERACTION_FIELDS,
    GID_SEQUENCE_NAMES,
    N_TAB_EMBEDDINGS,
    N_TIME_GAP_BUCKETS,
    POSITIVE_TARGET_FIELDS,
    SOURCE_BEHAVIOR_FIELDS,
    assign_day_split,
    day_split_boundaries,
    encode_category_with_unknown,
    recent_behavior_bounds,
    validate_gid_sequence_lengths,
)
from .user_features import KuaiRandUserFeatureStore
from ..sid.artifact import SemanticIDArtifact
from ..sid.token_codec import SidTokenCodec


LOG_COLUMNS = (
    "user_id",
    "video_id",
    "hourmin",
    "time_ms",
    *SOURCE_BEHAVIOR_FIELDS,
    "duration_ms",
    "tab",
)

LOG_DTYPES = {
    "user_id": np.int32,
    "video_id": np.int32,
    "hourmin": np.int16,
    "time_ms": np.int64,
    **{field: np.int8 for field in SOURCE_BEHAVIOR_FIELDS},
    "duration_ms": np.int32,
    "tab": np.int8,
}

TIME_GAP_BINS_SECONDS = np.asarray(
    [60, 300, 1_800, 21_600, 86_400, 604_800, 2_592_000]
)
EVENT_TIMEZONE = "Asia/Shanghai"


def _local_calendar_from_time_ms(
    time_ms: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    local_time = pd.to_datetime(
        time_ms,
        unit="ms",
        utc=True,
    ).dt.tz_convert(EVENT_TIMEZONE)
    date = (
        local_time.dt.year * 10_000
        + local_time.dt.month * 100
        + local_time.dt.day
    ).astype(np.int32)
    day_of_week = local_time.dt.dayofweek.astype(np.int8)
    return date, day_of_week


def _positive_target_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame.loc[:, list(POSITIVE_TARGET_FIELDS)].eq(1).any(axis=1)
        & ~frame["is_hate"].eq(1)
    ).to_numpy(dtype=np.bool_)


@dataclass
class UserExposureSequence:
    user_feature_row: int
    gid_ids: np.ndarray
    time_ms: np.ndarray
    dates: np.ndarray
    hour: np.ndarray
    day_of_week: np.ndarray
    time_gap_bucket: np.ndarray
    tab: np.ndarray
    behavior_time_ms: Dict[str, np.ndarray]
    behavior_gid_ids: Dict[str, np.ndarray]
    long_view_duration_bucket: np.ndarray
    positive_target: np.ndarray


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

    logs["date"], logs["request_day_of_week"] = (
        _local_calendar_from_time_ms(logs["time_ms"])
    )

    duration_seconds = logs["duration_ms"].clip(lower=0) / 1000.0
    logs["long_view_duration_bucket"] = np.minimum(
        np.rint(np.sqrt(duration_seconds)),
        99,
    ).astype(np.int8)

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
            gid_ids = group["video_id"].to_numpy(np.int32) + 1
            time_ms = group["time_ms"].to_numpy(np.int64)
            behavior_masks = {
                "click": group["is_click"].eq(1).to_numpy(),
                "long_view": group["long_view"].eq(1).to_numpy(),
                "like": group["is_like"].eq(1).to_numpy(),
                "deep_interact": group[list(DEEP_INTERACTION_FIELDS)]
                .eq(1)
                .any(axis=1)
                .to_numpy(),
                "hate": group["is_hate"].eq(1).to_numpy(),
            }
            behavior_positions = {
                name: np.flatnonzero(behavior_masks[name]).astype(np.int32)
                for name in GID_SEQUENCE_NAMES
            }
            behavior_gid_ids = {
                name: gid_ids[behavior_positions[name]]
                for name in GID_SEQUENCE_NAMES
            }
            behavior_time_ms = {
                name: time_ms[behavior_positions[name]]
                for name in GID_SEQUENCE_NAMES
            }
            positive_target = _positive_target_mask(group)
            user_feature_row = self.user_features.row_for_user(int(user_id))
            if user_feature_row == 0:
                raise ValueError(
                    f"user_id={int(user_id)} is missing from user_features"
                )
            self.sequences.append(
                UserExposureSequence(
                    user_feature_row=user_feature_row,
                    gid_ids=gid_ids,
                    time_ms=time_ms,
                    dates=group["date"].to_numpy(np.int32),
                    hour=group["request_hour"].to_numpy(np.int8),
                    day_of_week=group[
                        "request_day_of_week"
                    ].to_numpy(np.int8),
                    time_gap_bucket=group[
                        "time_gap_bucket"
                    ].to_numpy(np.int8),
                    tab=group["tab"].fillna(0).to_numpy(np.int8),
                    behavior_time_ms=behavior_time_ms,
                    behavior_gid_ids=behavior_gid_ids,
                    long_view_duration_bucket=group[
                        "long_view_duration_bucket"
                    ].to_numpy(np.int8)[behavior_positions["long_view"]],
                    positive_target=positive_target,
                )
            )


def build_exposure_sample_indices(
    corpus: KuaiRandExposureCorpus,
    sid_artifact: SemanticIDArtifact,
    min_history: int,
    warmup_days: int = 3,
    test_days: int = 3,
) -> Dict[str, ExposureSampleIndex]:
    """Split exposures by natural day.

    The first ``warmup_days`` dates provide history only and yield no samples.
    The last ``test_days`` dates form the test split; the remaining middle
    dates form the training split.
    """
    if min_history < 0:
        raise ValueError("min_history must be non-negative")
    if warmup_days < 0 or test_days < 0:
        raise ValueError("warmup_days and test_days must be non-negative")

    warmup_dates, test_dates = day_split_boundaries(
        (int(date) for sequence in corpus.sequences for date in sequence.dates),
        warmup_days=warmup_days,
        test_days=test_days,
    )

    indices = {
        split: ExposureSampleIndex.empty() for split in ("train", "test")
    }
    for sequence_index, sequence in enumerate(corpus.sequences):
        for position in range(min_history, len(sequence.gid_ids)):
            if not bool(sequence.positive_target[position]):
                continue
            target_time = int(sequence.time_ms[position])
            if (
                min_history > 0
                and int(sequence.time_ms[min_history - 1]) >= target_time
            ):
                continue
            target_video_id = str(int(sequence.gid_ids[position]) - 1)
            if target_video_id not in sid_artifact.item_codes:
                continue
            date = int(sequence.dates[position])
            split = assign_day_split(date, warmup_dates, test_dates)
            if split is None:
                continue
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
        history_lengths: Dict[str, int] | None = None,
        sample: int = -1,
        seed: int = 42,
    ):
        self.sid_artifact = sid_artifact
        self.sid_codec = SidTokenCodec(sid_artifact.codebook_sizes)
        self.history_lengths = dict(
            DEFAULT_GID_SEQUENCE_LENGTHS
            if history_lengths is None
            else history_lengths
        )
        validate_gid_sequence_lengths(self.history_lengths)
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
            "target_input_ids": self.sid_codec.decoder_inputs(target_codes),
            "labels": self.sid_codec.decoder_labels(target_codes),
        }
        target_time = int(sequence.time_ms[target_position])
        for name in GID_SEQUENCE_NAMES:
            event_times = sequence.behavior_time_ms[name]
            start, end = recent_behavior_bounds(
                event_times,
                target_time=target_time,
                max_history=self.history_lengths[name],
            )
            example[f"{name}_gid_ids"] = sequence.behavior_gid_ids[name][
                start:end
            ].tolist()
            if name == "long_view":
                example["long_view_duration_bucket"] = (
                    sequence.long_view_duration_bucket[start:end].tolist()
                )
        return example


class KuaiRandCollator:
    """Pad independent behavior histories to a fixed per-stream length."""

    def __init__(self, history_lengths: Dict[str, int] | None = None):
        self.history_lengths = dict(
            DEFAULT_GID_SEQUENCE_LENGTHS
            if history_lengths is None
            else history_lengths
        )
        validate_gid_sequence_lengths(self.history_lengths)

    def __call__(
        self,
        features: List[Dict[str, Sequence[int] | Sequence[float]]],
    ) -> Dict[str, torch.Tensor]:
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
        for name in GID_SEQUENCE_NAMES:
            gid_field = f"{name}_gid_ids"
            padded_gid_ids = []
            masks = []
            max_length = self.history_lengths[name]
            for feature in features:
                values = list(feature[gid_field])[-max_length:]
                padding = max_length - len(values)
                padded_gid_ids.append(values + [0] * padding)
                masks.append([1] * len(values) + [0] * padding)
            batch[gid_field] = torch.tensor(padded_gid_ids, dtype=torch.long)
            batch[f"{name}_attention_mask"] = torch.tensor(
                masks,
                dtype=torch.long,
            )

        padded_duration = []
        duration_length = self.history_lengths["long_view"]
        for feature in features:
            values = list(feature["long_view_duration_bucket"])[
                -duration_length:
            ]
            padded_duration.append(
                values + [0] * (duration_length - len(values))
            )
        batch["long_view_duration_bucket"] = torch.tensor(
            padded_duration,
            dtype=torch.long,
        )
        batch["long_view_duration_attention_mask"] = batch[
            "long_view_attention_mask"
        ].clone()
        batch["target_input_ids"] = torch.tensor(
            [feature["target_input_ids"] for feature in features],
            dtype=torch.long,
        )
        batch["labels"] = torch.tensor(
            [feature["labels"] for feature in features],
            dtype=torch.long,
        )
        return batch
