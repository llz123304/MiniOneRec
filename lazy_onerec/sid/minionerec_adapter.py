"""Convert MiniOneRec token-form indices to the local numeric SID artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .artifact import SemanticIDArtifact


TOKEN_PATTERN = re.compile(r"<[a-z]_(\d+)>")


def _parse_code(value: Any) -> int:
    if isinstance(value, int):
        return value
    match = TOKEN_PATTERN.fullmatch(str(value))
    if not match:
        raise ValueError(f"unsupported MiniOneRec SID token: {value!r}")
    return int(match.group(1))


def load_minionerec_index(path: str | Path) -> Dict[str, List[int]]:
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    return {
        str(item_id): [_parse_code(token) for token in tokens]
        for item_id, tokens in payload.items()
    }


def convert_minionerec_index(
    input_path: str | Path,
    output_path: str | Path,
    codebook_sizes: Sequence[int] | None = None,
) -> SemanticIDArtifact:
    item_codes = load_minionerec_index(input_path)
    n_levels = len(next(iter(item_codes.values())))
    maxima = [0] * n_levels
    for codes in item_codes.values():
        if len(codes) != n_levels:
            raise ValueError("MiniOneRec index contains variable-length SIDs")
        for level, code in enumerate(codes):
            maxima[level] = max(maxima[level], code)

    if codebook_sizes is None:
        codebook_sizes = [maximum + 1 for maximum in maxima]
    if len(codebook_sizes) != n_levels:
        raise ValueError(
            f"received {len(codebook_sizes)} sizes for {n_levels} SID levels"
        )

    artifact = SemanticIDArtifact(
        codebook_sizes=tuple(codebook_sizes),
        item_codes={item_id: tuple(codes) for item_id, codes in item_codes.items()},
        metadata={
            "source": "MiniOneRec",
            "source_index": str(input_path),
        },
    )
    artifact.save(output_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codebook-sizes", type=int, nargs="+")
    args = parser.parse_args()
    artifact = convert_minionerec_index(
        args.input, args.output, args.codebook_sizes
    )
    print(
        f"saved {artifact.n_items} items; sizes={list(artifact.codebook_sizes)}; "
        f"collision_rate={artifact.collision_rate:.6f}"
    )


if __name__ == "__main__":
    main()
