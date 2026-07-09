#!/usr/bin/env python3
"""Expand Hugging Face ImageNet-1K Parquet shards for torchvision.ImageNet."""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


WNID_RE = re.compile(r"n\d{8}")


def convert_shard(source: str, output_root: str, split: str) -> tuple[str, int, int]:
    source_path = Path(source)
    root = Path(output_root)
    marker_dir = root / ".conversion_markers"
    marker = marker_dir / f"{source_path.name}.done"
    if marker.exists():
        return source_path.name, 0, 0

    destination_split = "val" if split == "validation" else "train"
    parquet = pq.ParquetFile(source_path)
    written = 0
    skipped = 0

    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["image", "label"])
        images = table.column("image").to_pylist()

        for image in images:
            original_name = Path(image["path"]).name
            matches = WNID_RE.findall(original_name)
            if not matches:
                raise ValueError(f"No WordNet synset in image path: {original_name}")
            wnid = matches[-1]
            class_dir = root / destination_split / wnid
            class_dir.mkdir(parents=True, exist_ok=True)
            destination = class_dir / original_name
            payload = image["bytes"]

            if destination.exists() and destination.stat().st_size == len(payload):
                skipped += 1
                continue

            temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                handle.write(payload)
            os.replace(temporary, destination)
            written += 1

    marker_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"written={written}\nskipped={skipped}\n", encoding="utf-8")
    return source_path.name, written, skipped


def build_meta(output_root: Path) -> None:
    import torch

    wnids = sorted(path.name for path in (output_root / "train").iterdir() if path.is_dir())
    if len(wnids) != 1000:
        raise RuntimeError(f"Expected 1000 training synsets, found {len(wnids)}")

    # torchvision only needs this mapping once train/ and val/ are already expanded.
    wnid_to_classes = {wnid: (wnid,) for wnid in wnids}
    torch.save((wnid_to_classes, []), output_root / "meta.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    shards: list[tuple[Path, str]] = []
    shards.extend((path, "train") for path in sorted(args.source.glob("train-*.parquet")))
    shards.extend(
        (path, "validation") for path in sorted(args.source.glob("validation-*.parquet"))
    )
    if len(shards) != 308:
        raise RuntimeError(f"Expected 308 Parquet shards, found {len(shards)}")

    args.output.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_skipped = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_shard, str(path), str(args.output), split): path.name
            for path, split in shards
        }
        for future in as_completed(futures):
            name, written, skipped = future.result()
            completed += 1
            total_written += written
            total_skipped += skipped
            print(
                f"[{completed:03d}/{len(shards)}] {name}: "
                f"written={written} skipped={skipped}",
                flush=True,
            )

    build_meta(args.output)
    print(
        f"CONVERSION COMPLETE: shards={completed} written={total_written} "
        f"skipped={total_skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
