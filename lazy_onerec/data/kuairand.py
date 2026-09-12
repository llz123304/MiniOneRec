"""KuaiRand sequential data adapter for next-SID generation.

History uses an independent GID sequence slot. Target GID is only used to look
up the semantic-ID label and is never inserted into its own history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..kuairand_schema import BINARY_EVENT_FIELDS
from ..sid.artifact import SemanticIDArtifact
from ..sid.token_codec import SidTokenCodec


LOG_COLUMNS = (
    "user_id",
    "video_id",
    "date",
    "time_ms",
    "is_click",
    *BINARY_EVENT_FIELDS,
    "play_time_ms",
    "duration_ms",
    "tab",
)

LOG_DTYPES = {
    "user_id": np.int32,
    "video_id": np.int32,
    "date": np.int32,
    "time_ms": np.int64,
    "is_click": np.int8,
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
class UserClickSequence:
    user_id: int
    gid_ids: np.ndarray
    dates: np.ndarray
    time_ms: np.ndarray
    binary: Dict[str, np.ndarray]
    play_ratio_bucket: np.ndarray
    time_gap_bucket: np.ndarray
    tab: np.ndarray


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
    logs = logs.loc[logs["is_click"].eq(1)].copy()
    logs.sort_values(["user_id", "time_ms"], inplace=True)

    # Remove only immediately repeated clicks; later revisits remain valid.
    repeated = (
        logs["user_id"].eq(logs["user_id"].shift())
        & logs["video_id"].eq(logs["video_id"].shift())
    )
    logs = logs.loc[~repeated].copy()

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
    return logs


class KuaiRandClickCorpus:
    """In-memory clicked-event sequences shared by all KuaiRand variants."""

    def __init__(
        self,
        data_root: str | Path,
    ):
        data_root = Path(data_root)
        data_dir = data_root / "data" if (data_root / "data").is_dir() else data_root
        logs = _read_standard_logs(data_dir)
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
        self.sequences: List[UserClickSequence] = []
        for _, group in logs.groupby("user_id", sort=False):
            group = group.reset_index(drop=True)
            self.sequences.append(
                UserClickSequence(
                    user_id=int(group["user_id"].iloc[0]),
                    gid_ids=group["video_id"].to_numpy(np.int32) + 1,
                    dates=group["date"].to_numpy(np.int32),
                    time_ms=group["time_ms"].to_numpy(np.int64),
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


class KuaiRandNextSidDataset(Dataset):
    """Rolling next-click SID samples from a shared KuaiRand corpus."""

    def __init__(
        self,
        corpus: KuaiRandClickCorpus,
        sid_artifact: SemanticIDArtifact,
        split: str,
        min_history: int = 3,
        max_history: int = 128,
        train_end_date: int = 20220421,
        valid_end_date: int = 20220429,
        sample: int = -1,
        seed: int = 42,
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"unknown split: {split}")
        self.split = split
        self.sid_artifact = sid_artifact
        self.sid_codec = SidTokenCodec(sid_artifact.codebook_sizes)
        self.min_history = min_history
        self.max_history = max_history
        self.train_end_date = train_end_date
        self.valid_end_date = valid_end_date
        self.sequences = corpus.sequences
        self.num_gid_embeddings = corpus.num_gid_embeddings
        self.samples: List[Tuple[int, int]] = []
        for sequence_index, sequence in enumerate(self.sequences):
            # Position is the target event; history is strictly [:position].
            for position in range(min_history, len(sequence.gid_ids)):
                date = int(sequence.dates[position])
                if not self._belongs_to_split(date):
                    continue
                target_video_id = str(int(sequence.gid_ids[position]) - 1)
                if target_video_id in sid_artifact.item_codes:
                    self.samples.append((sequence_index, position))

        if sample > 0 and sample < len(self.samples):
            rng = np.random.RandomState(seed)
            chosen = rng.choice(len(self.samples), size=sample, replace=False)
            self.samples = [self.samples[index] for index in chosen.tolist()]

    def _belongs_to_split(self, date: int) -> bool:
        if self.split == "train":
            return date <= self.train_end_date
        if self.split == "valid":
            return self.train_end_date < date <= self.valid_end_date
        return date > self.valid_end_date

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Sequence[int]]:
        sequence_index, target_position = self.samples[index]
        sequence = self.sequences[sequence_index]
        start = max(0, target_position - self.max_history)
        history_slice = slice(start, target_position)

        target_gid = int(sequence.gid_ids[target_position])
        target_video_id = str(target_gid - 1)
        target_codes = self.sid_artifact.item_codes[target_video_id]

        example: Dict[str, Sequence[int]] = {
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
        self, features: List[Dict[str, Sequence[int]]]
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
