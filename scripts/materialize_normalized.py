#!/usr/bin/env python3
"""
Copy approved normalized Flux variants into a flat finalized directory.

Input:
    data/normalized_multi/{cluster}/{image_stem}/{selected_key}.jpg
    data/normalized_multi/{cluster}_selections.json

Output:
    data/normalized_flux/{cluster}/{image_stem}.jpg

Selections marked "MANUAL" are reported and skipped.
Existing destination files are preserved unless --overwrite is supplied.

Usage:
    # Process one cluster
    python scripts/materialize_normalized.py --cluster pilaf_cl

    # Process all discovered clusters
    python scripts/materialize_normalized.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

MANUAL_VALUE = "MANUAL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize approved Flux variants into a flat directory."
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help="Cluster ID, e.g. pilaf_cl. If omitted, process all discovered clusters.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/normalized_multi",
        help="Variants/selections root (default: data/normalized_multi)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/normalized_flux",
        help="Finalized image root (default: data/normalized_flux)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination images that already exist",
    )
    return parser.parse_args()


def load_selections(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Selections JSON not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Selections JSON must be an object mapping image stems to selected keys."
        )

    return data


def selected_source(
    cluster_dir: Path,
    image_stem: str,
    selected_key: str,
) -> Path:
    filename = (
        selected_key
        if selected_key.lower().endswith(".jpg")
        else f"{selected_key}.jpg"
    )
    return cluster_dir / image_stem / filename


def discover_clusters(input_root: Path) -> list[str]:
    clusters = []
    for path in sorted(input_root.glob("*_selections.json")):
        cluster_id = path.name[: -len("_selections.json")]
        if cluster_id:
            clusters.append(cluster_id)
    return clusters


def process_cluster(
    cluster_id: str,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
) -> tuple[bool, dict[str, int]]:
    selections_path = input_root / f"{cluster_id}_selections.json"
    source_cluster_dir = input_root / cluster_id
    destination_cluster_dir = output_root / cluster_id

    try:
        selections = load_selections(selections_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {cluster_id}: {exc}", file=sys.stderr)
        return False, {}

    if not source_cluster_dir.is_dir():
        print(
            f"[ERROR] {cluster_id}: cluster variants directory not found: "
            f"{source_cluster_dir}",
            file=sys.stderr,
        )
        return False, {}

    automatic: list[tuple[str, str, Path, Path]] = []
    manual: list[str] = []
    errors: list[str] = []

    for raw_stem, raw_selection in sorted(selections.items()):
        image_stem = str(raw_stem).strip()

        if not image_stem:
            errors.append("Found an empty image-stem key.")
            continue

        if raw_selection is None:
            errors.append(f"{image_stem}: selection is null.")
            continue

        selected_key = str(raw_selection).strip()

        if not selected_key:
            errors.append(f"{image_stem}: selection is empty.")
            continue

        if selected_key.upper() == MANUAL_VALUE:
            manual.append(image_stem)
            continue

        source = selected_source(source_cluster_dir, image_stem, selected_key)
        destination = destination_cluster_dir / f"{image_stem}.jpg"

        if not source.is_file():
            errors.append(f"{image_stem}: selected file does not exist: {source}")
            continue

        automatic.append((image_stem, selected_key, source, destination))

    if errors:
        print(
            f"[ERROR] {cluster_id}: validation failed. No files were copied:",
            file=sys.stderr,
        )
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            f"Summary: automatic={len(automatic)}, manual={len(manual)}, "
            f"errors={len(errors)}",
            file=sys.stderr,
        )
        return False, {}

    destination_cluster_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_existing = 0

    print(f"\n=== Cluster: {cluster_id} ===")
    for image_stem, selected_key, source, destination in automatic:
        if destination.exists() and not overwrite:
            print(f"[SKIP] {image_stem}: destination already exists: {destination}")
            skipped_existing += 1
            continue

        shutil.copy2(source, destination)
        print(f"[COPY] {image_stem}: {selected_key} -> {destination}")
        copied += 1

    for image_stem in manual:
        manual_destination = destination_cluster_dir / f"{image_stem}.jpg"
        print(
            f"[MANUAL] {image_stem}: skipped; manually place the final image at "
            f"{manual_destination}"
        )

    print("Materialization complete for cluster.")
    print(f"  Cluster:              {cluster_id}")
    print(f"  Automatic selections: {len(automatic)}")
    print(f"  Copied:               {copied}")
    print(f"  Existing skipped:     {skipped_existing}")
    print(f"  Manual skipped:       {len(manual)}")
    print(f"  Errors:               0")
    print(f"  Output directory:     {destination_cluster_dir}")

    return True, {
        "automatic": len(automatic),
        "copied": copied,
        "skipped_existing": skipped_existing,
        "manual": len(manual),
        "errors": 0,
    }


def main() -> int:
    args = parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if args.cluster:
        clusters = [args.cluster]
    else:
        clusters = discover_clusters(input_root)
        if not clusters:
            print(
                f"[ERROR] No *_selections.json files found under {input_root}",
                file=sys.stderr,
            )
            return 1

    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")
    print(f"Overwrite  : {args.overwrite}")
    print(f"Clusters   : {', '.join(clusters)}")

    succeeded = 0
    failed = 0
    totals = {
        "automatic": 0,
        "copied": 0,
        "skipped_existing": 0,
        "manual": 0,
        "errors": 0,
    }

    for cluster_id in clusters:
        ok, stats = process_cluster(
            cluster_id=cluster_id,
            input_root=input_root,
            output_root=output_root,
            overwrite=args.overwrite,
        )

        if ok:
            succeeded += 1
            for key in totals:
                totals[key] += stats.get(key, 0)
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("Overall summary")
    print("=" * 60)
    print(f"  Clusters succeeded:   {succeeded}")
    print(f"  Clusters failed:      {failed}")
    print(f"  Automatic selections: {totals['automatic']}")
    print(f"  Copied:               {totals['copied']}")
    print(f"  Existing skipped:     {totals['skipped_existing']}")
    print(f"  Manual skipped:       {totals['manual']}")
    print(f"  Errors:               {totals['errors']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
