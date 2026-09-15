"""Shared KuaiRand event-field and vocabulary definitions."""

import math
from bisect import bisect_left
from typing import Iterable, Optional, Sequence, Set, Tuple

SOURCE_BEHAVIOR_FIELDS = (
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_profile_enter",
    "is_hate",
)

DEEP_INTERACTION_FIELDS = (
    "is_follow",
    "is_comment",
    "is_forward",
    "is_profile_enter",
)

POSITIVE_TARGET_FIELDS = (
    "is_click",
    "long_view",
    "is_like",
    *DEEP_INTERACTION_FIELDS,
)

GID_SEQUENCE_NAMES = (
    "click",
    "long_view",
    "like",
    "deep_interact",
    "hate",
)

DEFAULT_GID_SEQUENCE_LENGTHS = {
    "click": 128,
    "long_view": 128,
    "like": 64,
    "deep_interact": 32,
    "hate": 16,
}

CONTEXT_SEQUENCE_NAMES = (
    "click",
    "long_view",
    "long_view_duration",
    "like",
    "deep_interact",
    "hate",
)

DEFAULT_QFORMER_QUERY_COUNTS = {
    "click": 16,
    "long_view": 16,
    "long_view_duration": 16,
    "like": 8,
    "deep_interact": 4,
    "hate": 2,
}

N_TAB_EMBEDDINGS = 15
N_TIME_GAP_BUCKETS = 8
N_LONG_VIEW_DURATION_BUCKETS = 100

REQUEST_CATEGORICAL_FIELDS = (
    "request_tab",
    "request_hour",
    "request_day_of_week",
    "request_time_gap_bucket",
)

# Index 0 is reserved for UNKNOWN; valid source values are shifted by one.
REQUEST_CATEGORICAL_CARDINALITIES = (
    N_TAB_EMBEDDINGS + 1,
    24 + 1,
    7 + 1,
    N_TIME_GAP_BUCKETS + 1,
)

N_USER_REQUEST_CONTEXT_TOKENS = 2
N_STATIC_CONTEXT_TOKENS = N_USER_REQUEST_CONTEXT_TOKENS


def encode_category_with_unknown(value: int, valid_size: int) -> int:
    """Shift a zero-based category by one and reserve zero for UNKNOWN."""
    return value + 1 if 0 <= value < valid_size else 0


def bucket_long_view_duration_ms(duration_ms: int) -> int:
    """Apply the configured square-root bucket to video duration."""
    seconds = max(int(duration_ms), 0) / 1000.0
    return min(round(math.sqrt(seconds)), 99)


def recent_behavior_bounds(
    event_times: Sequence[int],
    target_time: int,
    max_history: int,
) -> Tuple[int, int]:
    """Return the last history window whose events strictly precede target."""
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    end = bisect_left(event_times, target_time)
    return max(0, end - max_history), end


def validate_gid_sequence_lengths(lengths: dict[str, int]) -> None:
    """Validate one positive history length for every GID sequence."""
    if set(lengths) != set(GID_SEQUENCE_NAMES):
        raise ValueError(
            f"sequence lengths must define exactly {GID_SEQUENCE_NAMES}"
        )
    if any(int(lengths[name]) <= 0 for name in GID_SEQUENCE_NAMES):
        raise ValueError("all sequence lengths must be positive")


def validate_qformer_query_counts(counts: dict[str, int]) -> None:
    """Validate one positive query count for every context sequence."""
    if set(counts) != set(CONTEXT_SEQUENCE_NAMES):
        raise ValueError(
            f"query counts must define exactly {CONTEXT_SEQUENCE_NAMES}"
        )
    if any(int(counts[name]) <= 0 for name in CONTEXT_SEQUENCE_NAMES):
        raise ValueError("all Q-Former query counts must be positive")


def raw_context_length(lengths: dict[str, int]) -> int:
    """Return context length before behavior-sequence compression."""
    validate_gid_sequence_lengths(lengths)
    return (
        N_STATIC_CONTEXT_TOKENS
        + sum(int(lengths[name]) for name in GID_SEQUENCE_NAMES)
        + int(lengths["long_view"])
    )


def total_context_length(
    lengths: dict[str, int],
    query_counts: dict[str, int] | None = None,
) -> int:
    """Return static-token and Q-Former-compressed context length."""
    validate_gid_sequence_lengths(lengths)
    counts = (
        DEFAULT_QFORMER_QUERY_COUNTS
        if query_counts is None
        else query_counts
    )
    validate_qformer_query_counts(counts)
    source_lengths = {
        **lengths,
        "long_view_duration": lengths["long_view"],
    }
    if any(
        int(counts[name]) > int(source_lengths[name])
        for name in CONTEXT_SEQUENCE_NAMES
    ):
        raise ValueError("Q-Former query count cannot exceed source length")
    return N_STATIC_CONTEXT_TOKENS + sum(
        int(counts[name]) for name in CONTEXT_SEQUENCE_NAMES
    )


def day_split_boundaries(
    dates: Iterable[int],
    warmup_days: int,
    test_days: int,
) -> Tuple[Set[int], Set[int]]:
    """Return (warmup_dates, test_dates) from all observed dates.

    The first ``warmup_days`` distinct dates are history-only; the last
    ``test_days`` distinct dates are the test split. Remaining dates train.
    """
    if warmup_days < 0 or test_days < 0:
        raise ValueError("warmup_days and test_days must be non-negative")
    ordered = sorted({int(date) for date in dates})
    if len(ordered) <= warmup_days + test_days:
        raise ValueError(
            f"{len(ordered)} distinct dates cannot satisfy "
            f"warmup_days={warmup_days} + test_days={test_days} with a "
            "non-empty training range"
        )
    warmup_dates = set(ordered[:warmup_days])
    test_dates = set(ordered[len(ordered) - test_days:])
    return warmup_dates, test_dates


def assign_day_split(
    date: int,
    warmup_dates: Set[int],
    test_dates: Set[int],
) -> Optional[str]:
    """Map one date to 'train'/'test', or None when it is history-only."""
    if date in warmup_dates:
        return None
    return "test" if date in test_dates else "train"
