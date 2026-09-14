"""Static KuaiRand user-feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ONEHOT_FIELDS = tuple(f"onehot_feat{index}" for index in range(18))
ONEHOT_MAX_VALUES = (
    1,
    6,
    49,
    1470,
    14,
    33,
    2,
    117,
    453,
    6,
    4,
    4,
    1,
    1,
    1,
    1,
    1,
    1,
)

CATEGORICAL_USER_FIELDS = (
    "user_id",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    *ONEHOT_FIELDS,
)

CONTINUOUS_USER_FIELDS = (
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
)

STRING_CATEGORICAL_FIELDS = (
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
)

UNKNOWN_VALUES = {"", "-124", "nan", "none", "null", "unknown"}


def _category_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return None if text.lower() in UNKNOWN_VALUES else text


def _encode_string_categories(
    values: pd.Series,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    normalized = [_category_text(value) for value in values]
    vocabulary = {
        value: index + 1
        for index, value in enumerate(
            sorted({value for value in normalized if value is not None})
        )
    }
    encoded = np.asarray(
        [vocabulary.get(value, 0) for value in normalized],
        dtype=np.int32,
    )
    return encoded, len(vocabulary) + 1, vocabulary


def _encode_bounded_integers(
    values: pd.Series,
    maximum: int,
    *,
    unknown_values: Tuple[int, ...] = (),
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    encoded = np.zeros(len(numeric), dtype=np.int32)
    valid = (
        np.isfinite(numeric)
        & (numeric == np.floor(numeric))
        & (numeric >= 0)
        & (numeric <= maximum)
    )
    for unknown in unknown_values:
        valid &= numeric != unknown
    encoded[valid] = numeric[valid].astype(np.int64) + 1
    return encoded


@dataclass
class KuaiRandUserFeatureStore:
    categorical: np.ndarray
    continuous: np.ndarray
    categorical_cardinalities: Tuple[int, ...]
    continuous_mean: np.ndarray
    continuous_std: np.ndarray
    user_id_to_row: np.ndarray
    string_vocabularies: Dict[str, Dict[str, int]]

    @classmethod
    def load(cls, path: str | Path) -> "KuaiRandUserFeatureStore":
        columns = [*CATEGORICAL_USER_FIELDS, *CONTINUOUS_USER_FIELDS]
        frame = pd.read_csv(path, usecols=columns)
        if frame.empty:
            raise ValueError("user_features is empty")
        user_ids = pd.to_numeric(frame["user_id"], errors="raise").to_numpy(
            np.int64
        )
        if np.any(user_ids < 0):
            raise ValueError("user_id must be non-negative")
        if len(np.unique(user_ids)) != len(user_ids):
            raise ValueError("user_features contains duplicate user_id values")

        order = np.argsort(user_ids, kind="stable")
        frame = frame.iloc[order].reset_index(drop=True)
        user_ids = user_ids[order]
        n_users = len(frame)

        categorical = np.zeros(
            (n_users + 1, len(CATEGORICAL_USER_FIELDS)),
            dtype=np.int32,
        )
        cardinalities = []
        vocabularies: Dict[str, Dict[str, int]] = {}

        categorical[1:, 0] = user_ids.astype(np.int32) + 1
        cardinalities.append(int(user_ids.max()) + 2)

        column = 1
        for field in ("user_active_degree",):
            encoded, cardinality, vocabulary = _encode_string_categories(
                frame[field]
            )
            categorical[1:, column] = encoded
            cardinalities.append(cardinality)
            vocabularies[field] = vocabulary
            column += 1

        for field in ("is_live_streamer", "is_video_author"):
            categorical[1:, column] = _encode_bounded_integers(
                frame[field],
                maximum=1,
                unknown_values=(-124,),
            )
            cardinalities.append(3)
            column += 1

        for field in STRING_CATEGORICAL_FIELDS[1:]:
            encoded, cardinality, vocabulary = _encode_string_categories(
                frame[field]
            )
            categorical[1:, column] = encoded
            cardinalities.append(cardinality)
            vocabularies[field] = vocabulary
            column += 1

        for field, maximum in zip(ONEHOT_FIELDS, ONEHOT_MAX_VALUES):
            categorical[1:, column] = _encode_bounded_integers(
                frame[field],
                maximum=maximum,
            )
            cardinalities.append(maximum + 2)
            column += 1
        if column != len(CATEGORICAL_USER_FIELDS):
            raise RuntimeError("user categorical feature order is inconsistent")

        continuous_raw = (
            frame.loc[:, CONTINUOUS_USER_FIELDS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(np.float32)
        )
        continuous_log = np.log1p(np.maximum(continuous_raw, 0.0))
        continuous_mean = continuous_log.mean(axis=0, dtype=np.float64).astype(
            np.float32
        )
        continuous_std = continuous_log.std(axis=0, dtype=np.float64).astype(
            np.float32
        )
        continuous_std = np.maximum(continuous_std, 1e-6)
        continuous = np.zeros(
            (n_users + 1, len(CONTINUOUS_USER_FIELDS)),
            dtype=np.float32,
        )
        continuous[1:] = (
            continuous_log - continuous_mean
        ) / continuous_std

        user_id_to_row = np.zeros(int(user_ids.max()) + 1, dtype=np.int32)
        user_id_to_row[user_ids] = np.arange(1, n_users + 1, dtype=np.int32)
        return cls(
            categorical=categorical,
            continuous=continuous,
            categorical_cardinalities=tuple(cardinalities),
            continuous_mean=continuous_mean,
            continuous_std=continuous_std,
            user_id_to_row=user_id_to_row,
            string_vocabularies=vocabularies,
        )

    def row_for_user(self, user_id: int) -> int:
        if user_id < 0 or user_id >= len(self.user_id_to_row):
            return 0
        return int(self.user_id_to_row[user_id])
