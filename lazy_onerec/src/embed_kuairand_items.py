"""Build KuaiRand item text and encode it with interchangeable models.

The script has two resumable stages:

1. Select the KuaiRand item catalog and stream-join caption/category metadata
   into a stable JSONL text cache.
2. Encode that cache in bounded batches and write an aligned NPY memmap.

The row contract is always:
    item_ids.npy[i] <-> item_texts.jsonl line i <-> item_embeddings.npy[i]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


CATEGORY_FIELDS = (
    "first_level_category_name",
    "second_level_category_name",
    "third_level_category_name",
    "fourth_level_category_name",
)
EMPTY_VALUES = {"", "nan", "none", "null", "unknown", "-124"}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    document_prefix: str = ""
    supports_mrl: bool = False


MODEL_PROFILES = (
    (
        "qwen3-embedding",
        ModelProfile("qwen3-embedding", supports_mrl=True),
    ),
    ("gte-qwen2", ModelProfile("gte-qwen2")),
    ("bge-m3", ModelProfile("bge-m3")),
    ("bge-large-zh", ModelProfile("bge-large-zh")),
    (
        "multilingual-e5",
        ModelProfile("multilingual-e5", document_prefix="passage: "),
    ),
    ("m3e", ModelProfile("m3e")),
)


def resolve_profile(model_name: str) -> ModelProfile:
    lowered = model_name.lower()
    for marker, profile in MODEL_PROFILES:
        if marker in lowered:
            return profile
    return ModelProfile("sentence-transformers-auto")


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def clean_value(value: object) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in EMPTY_VALUES else text


def unique_values(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_item_text(
    caption_row: Optional[Dict[str, str]],
    category_row: Optional[Dict[str, str]],
    basic_row: Optional[Dict[str, str]],
) -> str:
    parts: List[str] = []
    caption = clean_value((caption_row or {}).get("caption"))
    cover = clean_value((caption_row or {}).get("show_cover_text"))
    if caption:
        parts.append(f"标题：{caption}")
    if cover:
        parts.append(f"封面文字：{cover}")

    categories = unique_values(
        [
            clean_value((category_row or {}).get(field))
            for field in CATEGORY_FIELDS
        ]
    )
    if categories:
        parts.append(f"内容类目：{' > '.join(categories)}")

    if not parts:
        basic = basic_row or {}
        fallback_fields = (
            ("视频类型", "video_type"),
            ("上传类型", "upload_type"),
            ("标签", "tag"),
            ("音乐类型", "music_type"),
        )
        for label, field in fallback_fields:
            value = clean_value(basic.get(field))
            if value:
                parts.append(f"{label}：{value}")

    return "\n".join(parts) if parts else "视频内容未知"


class SortedCsvLookup:
    """Single-pass lookup for a CSV sorted by its integer id column."""

    def __init__(self, path: Path, id_column: str):
        self.path = path
        self.id_column = id_column
        self.handle = path.open(encoding="utf-8", newline="")
        self.reader = csv.DictReader(self.handle)
        if self.reader.fieldnames is None or id_column not in self.reader.fieldnames:
            self.handle.close()
            raise ValueError(f"{path} does not contain {id_column!r}")
        self.current: Optional[Dict[str, str]] = None
        self.current_id: Optional[int] = None
        self.previous_id = -1
        self._advance()

    def _advance(self) -> None:
        try:
            row = next(self.reader)
        except StopIteration:
            self.current = None
            self.current_id = None
            return
        current_id = int(row[self.id_column])
        if current_id < self.previous_id:
            raise ValueError(f"{self.path} is not sorted by {self.id_column}")
        self.previous_id = current_id
        self.current = row
        self.current_id = current_id

    def get(self, target_id: int) -> Optional[Dict[str, str]]:
        while self.current_id is not None and self.current_id < target_id:
            self._advance()
        if self.current_id == target_id:
            result = self.current
            self._advance()
            return result
        return None

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "SortedCsvLookup":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern!r} under {directory}, "
            f"found {matches}"
        )
    return matches[0]


def catalog_video_ids(catalog_path: Path) -> np.ndarray:
    values = pd.read_csv(
        catalog_path,
        usecols=["video_id"],
        dtype={"video_id": np.int32},
    )["video_id"].to_numpy(np.int32)
    return np.unique(values)


def clicked_video_ids(
    data_dir: Path, catalog_ids: np.ndarray
) -> np.ndarray:
    max_video_id = int(catalog_ids.max())
    selected = np.zeros(max_video_id + 1, dtype=np.bool_)
    for path in sorted(data_dir.glob("log_standard_*_*.csv")):
        chunks = pd.read_csv(
            path,
            usecols=["video_id", "is_click"],
            dtype={"video_id": np.int32, "is_click": np.int8},
            chunksize=500_000,
        )
        for chunk in tqdm(
            chunks,
            desc=f"Scanning {path.name}",
            unit="chunk",
            dynamic_ncols=True,
        ):
            ids = chunk.loc[chunk["is_click"].eq(1), "video_id"].to_numpy(
                np.int64
            )
            ids = ids[(ids >= 0) & (ids <= max_video_id)]
            selected[ids] = True
    result = np.flatnonzero(selected).astype(np.int32)
    catalog_mask = np.zeros(max_video_id + 1, dtype=np.bool_)
    catalog_mask[catalog_ids] = True
    return result[catalog_mask[result]]


def source_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.write("\n")
    os.replace(temporary, path)


def count_lines(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for _ in source)


def prepare_text_cache(args: argparse.Namespace) -> Tuple[Path, Path]:
    data_dir = Path(args.data_root) / "data"
    catalog_path = find_one(data_dir, "video_features_basic_*.csv")
    captions_path = Path(args.captions)
    categories_path = Path(args.categories)
    work_dir = Path(
        args.work_dir
        or f"lazy_onerec/output/kuairand_items/{args.scope}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    ids_path = work_dir / "item_ids.npy"
    texts_path = work_dir / "item_texts.jsonl"
    manifest_path = work_dir / "text_manifest.json"

    expected_sources = {
        "catalog": source_signature(catalog_path),
        "captions": source_signature(captions_path),
        "categories": source_signature(categories_path),
    }
    if (
        ids_path.exists()
        and texts_path.exists()
        and manifest_path.exists()
        and not args.rebuild_texts
    ):
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
        ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
        if (
            manifest.get("scope") != args.scope
            or manifest.get("limit") != args.limit
            or manifest.get("sources") != expected_sources
            or int(manifest.get("num_items", -1)) != len(ids)
            or count_lines(texts_path) != len(ids)
        ):
            raise ValueError(
                "existing text cache does not match current inputs; "
                "pass --rebuild-texts"
            )
        print(f"reuse text cache: items={len(ids)} path={work_dir}")
        return ids_path, texts_path

    ids = catalog_video_ids(catalog_path)
    if args.scope == "clicked":
        ids = clicked_video_ids(data_dir, ids)
    if args.limit is not None:
        ids = ids[: args.limit]
    if len(ids) == 0:
        raise ValueError("selected item catalog is empty")

    temporary_texts = texts_path.with_suffix(".jsonl.tmp")
    stats = {
        "caption": 0,
        "cover": 0,
        "category": 0,
        "fallback": 0,
    }
    with (
        SortedCsvLookup(captions_path, "final_video_id") as captions,
        SortedCsvLookup(categories_path, "final_video_id") as categories,
        SortedCsvLookup(catalog_path, "video_id") as basics,
        temporary_texts.open("w", encoding="utf-8") as target,
    ):
        for item_id_value in tqdm(
            ids,
            total=len(ids),
            desc="Preparing item text",
            unit="item",
            dynamic_ncols=True,
        ):
            item_id = int(item_id_value)
            caption_row = captions.get(item_id)
            category_row = categories.get(item_id)
            basic_row = basics.get(item_id)
            text = build_item_text(caption_row, category_row, basic_row)
            has_caption = bool(
                clean_value((caption_row or {}).get("caption"))
            )
            has_cover = bool(
                clean_value((caption_row or {}).get("show_cover_text"))
            )
            has_category = any(
                clean_value((category_row or {}).get(field))
                for field in CATEGORY_FIELDS
            )
            if has_caption:
                stats["caption"] += 1
            if has_cover:
                stats["cover"] += 1
            if has_category:
                stats["category"] += 1
            if not (has_caption or has_cover or has_category):
                stats["fallback"] += 1
            target.write(
                json.dumps(
                    {"item_id": item_id, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )

    np.save(ids_path, ids.astype(np.int32, copy=False))
    os.replace(temporary_texts, texts_path)
    atomic_json(
        manifest_path,
        {
            "scope": args.scope,
            "limit": args.limit,
            "num_items": len(ids),
            "sources": expected_sources,
            "coverage": stats,
            "template": (
                "标题 + 封面文字 + "
                "一级/二级/三级/四级内容类目；缺失时回退基础属性"
            ),
        },
    )
    print(f"prepared text cache: items={len(ids)} path={work_dir}")
    return ids_path, texts_path


def iter_text_batches(
    path: Path,
    item_ids: np.ndarray,
    start: int,
    batch_size: int,
    prefix: str,
) -> Iterator[Tuple[int, List[str]]]:
    batch: List[str] = []
    batch_start = start
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index < start:
                continue
            row = json.loads(line)
            expected = int(item_ids[index])
            if int(row["item_id"]) != expected:
                raise ValueError(
                    f"text/item order mismatch at {index}: "
                    f"{row['item_id']} != {expected}"
                )
            batch.append(prefix + row["text"])
            if len(batch) == batch_size:
                yield batch_start, batch
                batch_start = index + 1
                batch = []
        if batch:
            yield batch_start, batch


def resolve_device(requested: str, torch: object) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(name: str, device: str, torch: object) -> object:
    if name == "auto":
        return torch.float32 if device == "cpu" else torch.float16
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def encode_items(
    args: argparse.Namespace, ids_path: Path, texts_path: Path
) -> None:
    if not args.model_name:
        raise ValueError("--model-name is required unless --prepare-only")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "install embedding dependencies with: "
            "uv pip install -r lazy_onerec/requirements.txt"
        ) from error

    profile = resolve_profile(args.model_name)
    output_dir = Path(
        args.output_dir
        or (
            f"lazy_onerec/output/embeddings/"
            f"{model_slug(args.model_name)}-{args.scope}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "item_embeddings.npy"
    output_ids_path = output_dir / "item_ids.npy"
    config_path = output_dir / "embedding_config.json"
    progress_path = output_dir / "progress.json"

    item_ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    device = resolve_device(args.device, torch)
    torch_dtype = resolve_torch_dtype(args.model_dtype, device, torch)
    model = SentenceTransformer(
        args.model_name,
        device=device,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    model.max_seq_length = args.max_length
    full_dim = int(model.get_sentence_embedding_dimension())
    output_dim = args.output_dim or full_dim
    if not 0 < output_dim <= full_dim:
        raise ValueError(
            f"output_dim must be in [1,{full_dim}], got {output_dim}"
        )
    if output_dim != full_dim and not profile.supports_mrl:
        print(
            "warning: this profile does not declare Matryoshka support; "
            "dimension truncation may reduce quality",
            file=sys.stderr,
        )

    config: Dict[str, object] = {
        "model_name": args.model_name,
        "revision": args.revision,
        "model_profile": asdict(profile),
        "pooling": "sentence-transformers model configuration",
        "scope": args.scope,
        "num_items": len(item_ids),
        "full_embedding_dim": full_dim,
        "output_dim": output_dim,
        "normalize": args.normalize,
        "max_length": args.max_length,
        "model_dtype": args.model_dtype,
        "storage_dtype": args.storage_dtype,
        "item_ids_source": str(ids_path.resolve()),
        "item_texts_source": str(texts_path.resolve()),
    }

    if args.overwrite:
        for path in (
            embeddings_path,
            output_ids_path,
            config_path,
            progress_path,
        ):
            path.unlink(missing_ok=True)

    start = 0
    if (
        embeddings_path.exists()
        or output_ids_path.exists()
        or config_path.exists()
        or progress_path.exists()
    ):
        if not all(
            path.exists()
            for path in (
                embeddings_path,
                output_ids_path,
                config_path,
                progress_path,
            )
        ):
            raise ValueError(
                f"incomplete output under {output_dir}; use --overwrite"
            )
        with config_path.open(encoding="utf-8") as source:
            existing_config = json.load(source)
        existing_config.pop("completed", None)
        if existing_config != config:
            raise ValueError(
                f"embedding config changed under {output_dir}; "
                "use a new --output-dir or --overwrite"
            )
        with progress_path.open(encoding="utf-8") as source:
            start = int(json.load(source)["next_index"])
        if not 0 <= start <= len(item_ids):
            raise ValueError(f"invalid resume position: {start}")
        output_ids = np.load(
            output_ids_path, mmap_mode="r", allow_pickle=False
        )
        if not np.array_equal(item_ids, output_ids):
            raise ValueError("output item_ids.npy differs from text cache")
        embeddings = np.load(
            embeddings_path, mmap_mode="r+", allow_pickle=False
        )
        if embeddings.shape != (len(item_ids), output_dim):
            raise ValueError(
                f"unexpected embedding shape {embeddings.shape}; "
                "use --overwrite"
            )
        print(f"resume encoding at item {start}/{len(item_ids)}")
    else:
        storage_dtype = np.dtype(args.storage_dtype)
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=storage_dtype,
            shape=(len(item_ids), output_dim),
        )
        np.save(output_ids_path, np.asarray(item_ids, dtype=np.int32))
        atomic_json(config_path, config)
        atomic_json(
            progress_path,
            {"next_index": 0, "num_items": len(item_ids)},
        )

    with tqdm(
        total=len(item_ids),
        initial=start,
        desc="Encoding items",
        unit="item",
        dynamic_ncols=True,
    ) as progress:
        for batch_start, texts in iter_text_batches(
            texts_path,
            item_ids,
            start,
            args.write_batch_size,
            profile.document_prefix,
        ):
            values = model.encode(
                texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32, copy=False)
            values = values[:, :output_dim]
            if args.normalize:
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                values = values / np.maximum(norms, 1e-12)
            batch_end = batch_start + len(values)
            embeddings[batch_start:batch_end] = values.astype(
                args.storage_dtype, copy=False
            )
            embeddings.flush()
            atomic_json(
                progress_path,
                {"next_index": batch_end, "num_items": len(item_ids)},
            )
            progress.update(len(values))

    config["completed"] = True
    atomic_json(config_path, config)
    print(
        f"embedding complete: shape={embeddings.shape} "
        f"dtype={embeddings.dtype} path={embeddings_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument(
        "--revision",
        help="optional Hugging Face model revision or commit hash",
    )
    parser.add_argument(
        "--data-root",
        default="lazy_onerec/KuaiRand-1K",
    )
    parser.add_argument(
        "--captions",
        default="lazy_onerec/KuaiRand-1K/kuairand_video_captions.csv",
    )
    parser.add_argument(
        "--categories",
        default="lazy_onerec/KuaiRand-1K/kuairand_video_categories.csv",
    )
    parser.add_argument(
        "--scope",
        choices=("clicked", "catalog"),
        default="clicked",
        help="clicked videos or the complete 1K video catalog",
    )
    parser.add_argument(
        "--work-dir",
        help=(
            "shared item-text cache; defaults to "
            "lazy_onerec/output/kuairand_items/<scope>"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--write-batch-size", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dim", type=int)
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-texts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="deterministic prefix limit for smoke tests only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.write_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    ids_path, texts_path = prepare_text_cache(args)
    if not args.prepare_only:
        encode_items(args, ids_path, texts_path)


if __name__ == "__main__":
    main()
