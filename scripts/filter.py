"""
Image quality filter for scraped food images.
Filters out low-resolution, corrupted, duplicate, poorly formatted,
and non-food images (using CLIP similarity).
Preserves term-slug folder structure from scrape.py output:
    Input:  data/raw/{cluster_id}/{term_slug}/
    Output: data/filtered/{cluster_id}/{term_slug}/
Usage:
    # Filter a specific cluster (basic filters)
    python scripts/filter.py --cluster pilaf_cl
    # Dry run
    python scripts/filter.py --cluster pilaf_cl --dry-run
"""
import os
import json
import argparse
import shutil
from pathlib import Path
from PIL import Image
import imagehash

# Filter thresholds
DEFAULT_MIN_SIZE = 512
DEFAULT_MIN_FILESIZE = 10240   # 10KB
DEFAULT_MAX_ASPECT_RATIO = 3.0
DEFAULT_HASH_THRESHOLD = 5
DEFAULT_CLIP_THRESHOLD = 0.2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def load_clusters(path="notes/clusters.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cluster_clip_query(cluster):
    if "clip_query_term" not in cluster:
        raise ValueError(
            f"Cluster '{cluster['cluster_id']}' is missing 'clip_query_term' field. "
            f"Please define it in clusters.json."
        )
    return f"a photo of {cluster['clip_query_term']}"


def load_clip_model():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = tokenizer
    return _clip_model, _clip_preprocess, _clip_tokenizer


def clip_similarity(image_path, text_query):
    import torch
    model, preprocess, tokenizer = load_clip_model()
    device = next(model.parameters()).device
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        text_tensor = tokenizer([text_query]).to(device)
        with torch.no_grad():
            img_features = model.encode_image(img_tensor)
            text_features = model.encode_text(text_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (img_features @ text_features.T).item()
        return similarity
    except Exception:
        return 0.0



def load_image_safe(path):
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        return img
    except Exception:
        return None


def get_image_hash(path):
    try:
        img = Image.open(path)
        return imagehash.phash(img)
    except Exception:
        return None


def filter_term_folder(input_dir, output_dir, min_size, min_filesize,
                        max_aspect_ratio, hash_threshold, seen_hashes,
                        dry_run=False, use_clip=False,
                        clip_query=None, clip_threshold=0.2):
    """
    Filter images in a single term-slug folder.
    seen_hashes is shared across term folders to catch cross-term duplicates.
    Returns stats dict.
    """
    stats = {
        "total": 0, "passed": 0, "too_small_res": 0,
        "too_small_file": 0, "bad_aspect": 0, "corrupted": 0,
        "duplicate": 0, "not_image": 0, "low_clip": 0
    }

    input_path = Path(input_dir)
    if not input_path.exists():
        return stats

    files = sorted([
        f for f in input_path.iterdir()
        if f.is_file()
    ])

    passed_count = 0

    for filepath in files:
        ext = filepath.suffix.lower()
        stats["total"] += 1

        if ext not in IMAGE_EXTENSIONS:
            stats["not_image"] += 1
            continue

        if filepath.stat().st_size < min_filesize:
            stats["too_small_file"] += 1
            continue

        img = load_image_safe(filepath)
        if img is None:
            stats["corrupted"] += 1
            continue

        w, h = img.size
        if w < min_size or h < min_size:
            stats["too_small_res"] += 1
            continue

        if max(w / h, h / w) > max_aspect_ratio:
            stats["bad_aspect"] += 1
            continue

        img_hash = get_image_hash(filepath)
        if img_hash is not None:
            is_dup = any(
                abs(img_hash - eh) < hash_threshold
                for eh in seen_hashes
            )
            if is_dup:
                stats["duplicate"] += 1
                continue

        if use_clip and clip_query:
            sim = clip_similarity(filepath, clip_query)
            if sim < clip_threshold:
                stats["low_clip"] += 1
                continue

        # Register the hash only after the image passes every acceptance
        # filter. A CLIP-rejected image must not block a later near-duplicate
        # that would otherwise be accepted.
        if img_hash is not None:
            seen_hashes.add(img_hash)

        # Passed all filters
        passed_count += 1
        stats["passed"] += 1

        if not dry_run:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_name = f"img_{passed_count:03d}{ext}"
            shutil.copy2(filepath, Path(output_dir) / out_name)

    return stats


def filter_cluster(cluster, raw_dir, output_dir, min_size, min_filesize,
                   max_aspect_ratio, hash_threshold, dry_run=False,
                   use_clip=False, clip_threshold=0.2,
                   term_filter=None):
    """Filter all term-slug folders for a cluster."""
    cluster_id = cluster["cluster_id"]
    clip_query = get_cluster_clip_query(cluster) if use_clip else None

    print(f"\n{'='*60}")
    print(f"Cluster: {cluster['cluster_name']}")
    if use_clip:
        print(f"CLIP query: '{clip_query}'")
    print(f"{'='*60}")

    cluster_raw = raw_dir / cluster_id
    if not cluster_raw.exists():
        print(f"  No raw images at {cluster_raw}")
        return

    # Get term-slug folders
    term_dirs = sorted([
        d for d in cluster_raw.iterdir()
        if d.is_dir()
    ])

    if term_filter:
        term_dirs = [d for d in term_dirs if d.name == term_filter]
        if not term_dirs:
            print(f"  Term '{term_filter}' not found in {cluster_raw}")
            return

    if not term_dirs:
        print(f"  No term folders found in {cluster_raw}")
        return

    # Shared hash set across ALL term folders — catches cross-term duplicates
    seen_hashes = set()
    total_stats = {
        "total": 0, "passed": 0, "too_small_res": 0,
        "too_small_file": 0, "bad_aspect": 0, "corrupted": 0,
        "duplicate": 0, "not_image": 0, "low_clip": 0
    }

    for term_dir in term_dirs:
        term_slug = term_dir.name
        out_dir = output_dir / cluster_id / term_slug

        stats = filter_term_folder(
            term_dir, out_dir,
            min_size=min_size,
            min_filesize=min_filesize,
            max_aspect_ratio=max_aspect_ratio,
            hash_threshold=hash_threshold,
            seen_hashes=seen_hashes,
            dry_run=dry_run,
            use_clip=use_clip,
            clip_query=clip_query,
            clip_threshold=clip_threshold,
        )

        clip_info = f" | Low CLIP: {stats['low_clip']}" if use_clip else ""
        print(f"\n  [{term_slug}]")
        print(f"    Total: {stats['total']} | Passed: {stats['passed']} | "
              f"Small: {stats['too_small_res']} | Bad aspect: {stats['bad_aspect']} | "
              f"Corrupted: {stats['corrupted']} | Duplicate: {stats['duplicate']}"
              f"{clip_info}")

        for key in total_stats:
            total_stats[key] += stats[key]

    print(f"\n  {'─'*50}")
    print(f"  CLUSTER TOTAL: {total_stats['total']} scanned → {total_stats['passed']} passed")
    return total_stats


def main():
    parser = argparse.ArgumentParser(description="Filter scraped food images")
    parser.add_argument("--clusters-file", default="notes/clusters.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/filtered")
    parser.add_argument("--cluster", default=None)
    parser.add_argument("--term", default=None,
                        help="Filter only this term-slug folder")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--min-filesize", type=int, default=DEFAULT_MIN_FILESIZE)
    parser.add_argument("--max-aspect-ratio", type=float, default=DEFAULT_MAX_ASPECT_RATIO)
    parser.add_argument("--hash-threshold", type=int, default=DEFAULT_HASH_THRESHOLD)
    parser.add_argument("--clip-threshold", type=float, default=DEFAULT_CLIP_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = load_clusters(args.clusters_file)
    clusters = data["clusters"]

    if args.cluster:
        clusters = [c for c in clusters if c["cluster_id"] == args.cluster]
        if not clusters:
            print(f"Cluster '{args.cluster}' not found.")
            return

    print("Loading CLIP model...")
    load_clip_model()
    print("CLIP model loaded.\n")

    if args.dry_run:
        print(
            "DRY RUN — no output directories will be created and no images "
            "will be copied or modified by this script.\n"
        )

    for cluster in clusters:
        filter_cluster(
            cluster,
            raw_dir=Path(args.raw_dir),
            output_dir=Path(args.output_dir),
            min_size=args.min_size,
            min_filesize=args.min_filesize,
            max_aspect_ratio=args.max_aspect_ratio,
            hash_threshold=args.hash_threshold,
            dry_run=args.dry_run,
            use_clip=True,
            clip_threshold=args.clip_threshold,
            term_filter=args.term,
        )

    if args.dry_run:
        print("\n\nDry run complete — project output files were not changed.")
    else:
        print(
            f"\n\nDone! Next step: manually review data/filtered/ "
            "before continuing the pipeline."
        )


if __name__ == "__main__":
    main()