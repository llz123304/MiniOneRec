"""Framework-neutral semantic-ID artifact.

The SID layer owns raw zero-based codebook indices only. Model token offsets,
special tokens, embeddings, and sequence features belong to downstream code.
"""

from __future__ import annotations

import json
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from tqdm.auto import tqdm


FORMAT_NAME = "semantic-id-index"
FORMAT_VERSION = 1


@dataclass
class SemanticIDArtifact:
    codebook_sizes: Tuple[int, ...]
    item_codes: Dict[str, Tuple[int, ...]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _normalized: InitVar[bool] = False

    def __post_init__(self, _normalized: bool) -> None:
        self.codebook_sizes = tuple(int(k) for k in self.codebook_sizes)
        if not _normalized:
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
        cached = self.metadata.get("sid_metrics_summary", {}).get(
            "collision_count"
        )
        if cached is not None:
            return int(cached)
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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("{\n")
            f.write(f'  "format": {json.dumps(FORMAT_NAME)},\n')
            f.write(f'  "version": {FORMAT_VERSION},\n')
            f.write(
                '  "codebook_sizes": '
                f"{json.dumps(list(self.codebook_sizes))},\n"
            )
            f.write(
                '  "metadata": '
                f"{json.dumps(self.metadata, ensure_ascii=False)},\n"
            )
            f.write('  "items": {\n')
            items = tqdm(
                self.item_codes.items(),
                total=self.n_items,
                desc="Saving SID index",
                unit="item",
                dynamic_ncols=True,
            )
            for index, (item_id, codes) in enumerate(items):
                f.write(f"    {json.dumps(item_id, ensure_ascii=False)}: ")
                f.write(json.dumps(list(codes)))
                f.write(",\n" if index + 1 < self.n_items else "\n")
            f.write("  }\n}\n")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticIDArtifact":
        with Path(path).open(encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != FORMAT_NAME:
            raise ValueError(f"unsupported SID format: {payload.get('format')}")
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported SID version: {payload.get('version')}")
        item_codes = payload["items"]
        for item_id, codes in item_codes.items():
            item_codes[item_id] = tuple(int(code) for code in codes)
        return cls(
            codebook_sizes=tuple(payload["codebook_sizes"]),
            item_codes=item_codes,
            metadata=payload.get("metadata", {}),
            _normalized=True,
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
            _normalized=True,
        )
