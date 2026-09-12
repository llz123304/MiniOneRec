"""Framework-neutral semantic-ID artifact.

The SID layer owns raw zero-based codebook indices only. Model token offsets,
special tokens, embeddings, and sequence features belong to downstream code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


FORMAT_NAME = "semantic-id-index"
FORMAT_VERSION = 1


@dataclass
class SemanticIDArtifact:
    codebook_sizes: Tuple[int, ...]
    item_codes: Dict[str, Tuple[int, ...]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.codebook_sizes = tuple(int(k) for k in self.codebook_sizes)
        self.item_codes = {
            str(item_id): tuple(int(code) for code in codes)
            for item_id, codes in self.item_codes.items()
        }
        self.validate()

    @property
    def n_levels(self) -> int:
        return len(self.codebook_sizes)

    @property
    def n_items(self) -> int:
        return len(self.item_codes)

    @property
    def collision_count(self) -> int:
        return self.n_items - len(set(self.item_codes.values()))

    @property
    def collision_rate(self) -> float:
        return self.collision_count / self.n_items

    def validate(self, require_unique: bool = False) -> None:
        if not self.codebook_sizes or any(k <= 0 for k in self.codebook_sizes):
            raise ValueError("codebook_sizes must contain positive integers")
        if not self.item_codes:
            raise ValueError("item_codes cannot be empty")
        for item_id, codes in self.item_codes.items():
            if len(codes) != self.n_levels:
                raise ValueError(
                    f"item {item_id!r} has {len(codes)} codes; "
                    f"expected {self.n_levels}"
                )
            for level, (code, size) in enumerate(zip(codes, self.codebook_sizes)):
                if not 0 <= code < size:
                    raise ValueError(
                        f"item {item_id!r}, level {level}: "
                        f"code {code} outside [0, {size})"
                    )
        if require_unique and self.collision_count:
            raise ValueError(f"artifact contains {self.collision_count} collided items")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "codebook_sizes": list(self.codebook_sizes),
            "metadata": dict(self.metadata),
            "items": {
                item_id: list(codes)
                for item_id, codes in self.item_codes.items()
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SemanticIDArtifact":
        with Path(path).open(encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != FORMAT_NAME:
            raise ValueError(f"unsupported SID format: {payload.get('format')}")
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported SID version: {payload.get('version')}")
        return cls(
            codebook_sizes=tuple(payload["codebook_sizes"]),
            item_codes=payload["items"],
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def from_rows(
        cls,
        item_ids: Iterable[Any],
        codes: Iterable[Sequence[int]],
        codebook_sizes: Sequence[int],
        metadata: Mapping[str, Any] | None = None,
    ) -> "SemanticIDArtifact":
        return cls(
            codebook_sizes=tuple(codebook_sizes),
            item_codes={
                str(item_id): tuple(int(code) for code in row)
                for item_id, row in zip(item_ids, codes)
            },
            metadata=dict(metadata or {}),
        )
