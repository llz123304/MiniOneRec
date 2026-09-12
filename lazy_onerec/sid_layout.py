"""Shared semantic-ID token layout."""

from typing import Sequence, Tuple


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
N_SPECIAL = 3


def sid_level_offsets(codebook_sizes: Sequence[int]) -> Tuple[int, ...]:
    offsets = []
    offset = N_SPECIAL
    for size in codebook_sizes:
        offsets.append(offset)
        offset += int(size)
    return tuple(offsets)
