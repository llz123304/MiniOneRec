"""Model-side token layout for raw semantic-ID codes."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .layout import (
    BOS_ID,
    EOS_ID,
    N_SPECIAL,
    PAD_ID,
    sid_level_offsets,
)


@dataclass(frozen=True)
class SidTokenCodec:
    """Map raw per-level SID codes to decoder token ids."""

    codebook_sizes: Tuple[int, ...]

    def __init__(self, codebook_sizes: Sequence[int]):
        sizes = tuple(int(size) for size in codebook_sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("codebook_sizes must contain positive integers")
        object.__setattr__(self, "codebook_sizes", sizes)

    @property
    def n_levels(self) -> int:
        return len(self.codebook_sizes)

    @property
    def level_offsets(self) -> Tuple[int, ...]:
        return sid_level_offsets(self.codebook_sizes)

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + sum(self.codebook_sizes)

    def code_to_token(self, level: int, code: int) -> int:
        if not 0 <= level < self.n_levels:
            raise ValueError(f"invalid SID level: {level}")
        if not 0 <= code < self.codebook_sizes[level]:
            raise ValueError(
                f"level {level} code {code} outside "
                f"[0, {self.codebook_sizes[level]})"
            )
        return self.level_offsets[level] + code

    def token_to_code(self, level: int, token: int) -> int:
        if not 0 <= level < self.n_levels:
            raise ValueError(f"invalid SID level: {level}")
        code = int(token) - self.level_offsets[level]
        if not 0 <= code < self.codebook_sizes[level]:
            raise ValueError(
                f"token {token} is outside SID level {level}"
            )
        return code

    def encode_codes(self, codes: Sequence[int]) -> List[int]:
        if len(codes) != self.n_levels:
            raise ValueError(
                f"received {len(codes)} codes; expected {self.n_levels}"
            )
        return [
            self.code_to_token(level, int(code))
            for level, code in enumerate(codes)
        ]

    def decoder_inputs(self, codes: Sequence[int]) -> List[int]:
        return [BOS_ID] + self.encode_codes(codes)

    def decoder_labels(self, codes: Sequence[int]) -> List[int]:
        return [-100] + self.encode_codes(codes)

    def decode_tokens(self, tokens: Sequence[int]) -> List[int]:
        if len(tokens) != self.n_levels:
            raise ValueError(
                f"received {len(tokens)} tokens; expected {self.n_levels}"
            )
        return [
            self.token_to_code(level, int(token))
            for level, token in enumerate(tokens)
        ]
