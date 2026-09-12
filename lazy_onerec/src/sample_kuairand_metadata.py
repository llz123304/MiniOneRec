"""Create aligned small samples from KuaiRand caption/category metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


def choose_video_ids(
    catalog_path: Path, sample_size: int, seed: int
) -> List[int]:
    catalog = pd.read_csv(
        catalog_path,
        usecols=["video_id"],
        dtype={"video_id": np.int32},
    )
    if sample_size > len(catalog):
        raise ValueError(f"sample_size={sample_size} exceeds catalog size")
    return sorted(
        catalog["video_id"]
        .sample(n=sample_size, random_state=seed)
        .astype(int)
        .tolist()
    )


def extract_rows(path: Path, video_ids: Set[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    max_video_id = max(video_ids)
    previous_id = -1
    sorted_input = True

    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            current_id = int(row["final_video_id"])
            if current_id < previous_id:
                sorted_input = False
            previous_id = current_id
            if current_id in video_ids:
                rows.append(row)
                if len(rows) == len(video_ids):
                    break
            if sorted_input and current_id > max_video_id:
                break
    return sorted(rows, key=lambda row: int(row["final_video_id"]))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty sample: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(field for field in row if field not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        default=(
            "lazy_onerec/KuaiRand-1K/"
            "data/video_features_basic_1k.csv"
        ),
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
        "--output-dir",
        default="lazy_onerec/KuaiRand-1K/samples",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    video_ids = choose_video_ids(
        Path(args.catalog), args.sample_size, args.seed
    )
    selected = set(video_ids)
    caption_rows = extract_rows(Path(args.captions), selected)
    category_rows = extract_rows(Path(args.categories), selected)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "kuairand_video_captions.sample.csv", caption_rows)
    write_rows(output / "kuairand_video_categories.sample.csv", category_rows)

    captions_by_id = {
        row["final_video_id"]: row for row in caption_rows
    }
    categories_by_id = {
        row["final_video_id"]: row for row in category_rows
    }
    joined = []
    for video_id in video_ids:
        key = str(video_id)
        row = {"final_video_id": key}
        row.update(
            {
                field: value
                for field, value in captions_by_id.get(key, {}).items()
                if field != "final_video_id"
            }
        )
        row.update(
            {
                field: value
                for field, value in categories_by_id.get(key, {}).items()
                if field != "final_video_id"
            }
        )
        joined.append(row)
    write_rows(output / "kuairand_video_metadata.sample.csv", joined)

    print(
        f"sampled={len(video_ids)} captions={len(caption_rows)} "
        f"categories={len(category_rows)} output={output}"
    )


if __name__ == "__main__":
    main()
