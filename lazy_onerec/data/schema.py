"""Shared KuaiRand event-field and vocabulary definitions."""

BINARY_EVENT_FIELDS = (
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)

N_TAB_EMBEDDINGS = 15
N_PLAY_RATIO_BUCKETS = 8
N_TIME_GAP_BUCKETS = 8

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


def encode_category_with_unknown(value: int, valid_size: int) -> int:
    """Shift a zero-based category by one and reserve zero for UNKNOWN."""
    return value + 1 if 0 <= value < valid_size else 0
