"""Numeric SID codec for the Lazy Decoder-Only recommender.

The repo's index.json stores each item as string tokens "<a_x><b_y><c_z>",
which exist because the baseline reuses a *pretrained* LLM vocabulary. The lazy
decoder is trained FROM SCRATCH with its own vocabulary, so we use a clean
integer layout instead: each codebook level occupies its own contiguous id
range, which lets the model tell which level a token belongs to.

Vocabulary layout (per-level codebook sizes K_0, K_1, ...):

    id 0            -> PAD
    id 1            -> BOS
    id 2            -> EOS
    3 .. 3+K_0-1              -> level-0 codes  (0..K_0-1)
    3+K_0 .. 3+K_0+K_1-1      -> level-1 codes  (0..K_1-1)
    ...
    vocab_size = 3 + sum(K_i)

Each level occupies its own contiguous id range (via a cumulative offset), so a
raw code value maps to different token ids across levels: no cross-level
collision and no string parsing. Levels may have different sizes.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List


PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
_N_SPECIAL = 3

_SID_RE = re.compile(r"<[a-z]_(\d+)>")  # matches "<a_12>" -> 12


@dataclass
class SidCodec:
    """Numeric SID codec with a per-level codebook size.

    codebook_sizes[i] is K for level i (RQ-VAE num_emb_list, e.g. [256,256,256]
    or an asymmetric [512,256,128]). Each level occupies its own contiguous id
    range starting right after the 3 special ids.
    """

    codebook_sizes: List[int]

    def __post_init__(self):
        assert len(self.codebook_sizes) >= 1
        # cumulative id offset where each level's range begins (after specials)
        self._offsets = [_N_SPECIAL]
        for k in self.codebook_sizes:
            self._offsets.append(self._offsets[-1] + k)

    @property
    def n_levels(self) -> int:
        return len(self.codebook_sizes)

    @property
    def vocab_size(self) -> int:
        return _N_SPECIAL + sum(self.codebook_sizes)

    def code_to_id(self, level: int, code: int) -> int:
        assert 0 <= level < self.n_levels, level
        assert 0 <= code < self.codebook_sizes[level], (level, code)
        return self._offsets[level] + code

    def id_to_code(self, token_id: int):
        """Inverse of code_to_id. Returns (level, code) or None for specials."""
        if token_id < _N_SPECIAL:
            return None
        for level in range(self.n_levels):
            if token_id < self._offsets[level + 1]:
                return level, token_id - self._offsets[level]
        raise ValueError(f"token id {token_id} out of range (vocab={self.vocab_size})")

    def codes_to_ids(self, codes: List[int]) -> List[int]:
        """[c0, c1, c2] (one item, one code per level) -> [id0, id1, id2]."""
        assert len(codes) == self.n_levels, (len(codes), self.n_levels)
        return [self.code_to_id(l, c) for l, c in enumerate(codes)]

    def target_ids(self, codes: List[int]) -> List[int]:
        """[BOS] + per-level ids  (the decoder target stream)."""
        return [BOS_ID] + self.codes_to_ids(codes)


def load_index_as_codes(index_path: str) -> Dict[str, List[int]]:
    """Read the repo's index.json ("<a_x><b_y><c_z>") -> {item_id: [x, y, z]}."""
    with open(index_path) as f:
        raw = json.load(f)
    out: Dict[str, List[int]] = {}
    for item_id, tokens in raw.items():
        codes = [int(_SID_RE.match(t).group(1)) for t in tokens]
        out[item_id] = codes
    return out


def build_codec_from_index(index_path: str, codebook_sizes: List[int] = None,
                           ) -> "tuple[SidCodec, Dict[str, List[int]]]":
    """Build a SidCodec for an existing index.json.

    codebook_sizes is a per-level HYPERPARAMETER list (RQ-VAE num_emb_list, e.g.
    [256, 256, 256] or an asymmetric [512, 256, 128]) and must match the values
    used when the SIDs were generated. Pass it explicitly to stay in sync.

    If left None, each level's size is INFERRED as (max code in that level) + 1.
    Inference can under-size a level when its largest code value is unused, so
    prefer passing the real values. A mismatch is reported as an assertion.

    Returns (codec, item->codes).
    """
    item_codes = load_index_as_codes(index_path)
    n_levels = len(next(iter(item_codes.values())))
    # largest code seen per level
    per_level_max = [0] * n_levels
    for codes in item_codes.values():
        for l, c in enumerate(codes):
            per_level_max[l] = max(per_level_max[l], c)

    if codebook_sizes is None:
        codebook_sizes = [m + 1 for m in per_level_max]
    else:
        assert len(codebook_sizes) == n_levels, (len(codebook_sizes), n_levels)
        for l, (k, m) in enumerate(zip(codebook_sizes, per_level_max)):
            assert k > m, f"level {l}: codebook_size={k} too small, index has code {m}"

    codec = SidCodec(codebook_sizes=list(codebook_sizes))
    return codec, item_codes
