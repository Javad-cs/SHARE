"""
Sample curated images by centroid similarity with a per-group selection cap.

Images below mean - (std_multiplier * standard deviation) are excluded.
The remaining images are ranked by cosine similarity to the chosen centroid.

Centroid methods:
  global   — normalized mean of all curated image embeddings
  balanced — normalized mean of per-group centroids, giving each
             endonym group equal weight [default]

Usage:
    python scripts/sample.py --cluster pilaf_cl --n 40 --max-per-endonym 8
    python scripts/sample.py --cluster pilaf_cl --n 40 --max-per-endonym 8 --std-multiplier 0.5
    python scripts/sample.py --cluster pilaf_cl --n 40 --max-per-endonym 8 --copy
"""

import argparse
import csv
import shutil
import numpy as np
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_saved_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

def load_embeddings(embeddings_path):
    data = np.load(embeddings_path, allow_pickle=True)
    return data["embeddings"], data["paths"].tolist()


def collect_kept_images(manual_dir, cluster_id):
    keep_path = Path(manual_dir) / cluster_id / "keep"
    log_path  = Path(manual_dir) / cluster_id / "filter_log.csv"

    path_to_endonym = {}
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["decision"] == "k":
                    path_to_endonym[row["output_path"]] = row["original_endonym"]

    images = []
    for f in sorted(keep_path.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        endonym = path_to_endonym.get(str(f))
        if not endonym:
            parts = f.stem.split("_", 1)
            endonym = parts[1] if len(parts) > 1 else "unknown"
        images.append((f.resolve(), endonym))
    return images


def compute_centroid(embeddings, endonyms, method):
    if method == "global":
        centroid = embeddings.mean(axis=0)
    elif method == "balanced":
        unique = sorted(set(endonyms))
        endonyms_arr = np.array(endonyms)
        per_endonym_centroids = []
        for e in unique:
            mask = endonyms_arr == e
            embs = embeddings[mask]
            c = embs.mean(axis=0)
            c /= np.linalg.norm(c)
            per_endonym_centroids.append(c)
        centroid = np.mean(per_endonym_centroids, axis=0)
    else:
        raise ValueError(f"Unknown centroid method: {method}")

    centroid /= np.linalg.norm(centroid)
    return centroid


def compute_threshold(sims, std_multiplier):
    """Exclude images more than std_multiplier standard deviations below mean."""
    return float(sims.mean() - std_multiplier * sims.std())


def select_central_diverse(embeddings, endonyms, paths, n,
                           max_per_endonym, centroid, threshold):
    sims = embeddings @ centroid

    # apply threshold filter
    above_mask = sims >= threshold
    filtered_embeddings = embeddings[above_mask]
    filtered_endonyms   = np.array(endonyms)[above_mask]
    filtered_paths      = np.array(paths)[above_mask]
    filtered_sims       = sims[above_mask]

    # rank by similarity to centroid
    ranked = np.argsort(filtered_sims)[::-1]

    selected_local = []
    endonym_counts = {}

    for idx in ranked:
        endonym = filtered_endonyms[idx]
        count = endonym_counts.get(endonym, 0)
        if count >= max_per_endonym:
            continue
        selected_local.append(idx)
        endonym_counts[endonym] = count + 1
        if len(selected_local) >= n:
            break

    # map back to original indices
    original_indices = np.where(above_mask)[0]
    selected_original = [original_indices[i] for i in selected_local]

    return selected_original, sims, threshold, above_mask.sum()



def copy_selected(paths, selected_indices, endonyms, cluster_id,
                  output_dir, n, max_per_endonym, centroid_method,
                  threshold_method, std_multiplier):
    label = (f"median" if threshold_method == "median"
             else f"std{std_multiplier}")
    dest = (Path(output_dir) /
            f"{cluster_id}_sample_n{n}_max{max_per_endonym}_{centroid_method}_{label}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    endonyms_arr = np.array(endonyms)
    for rank, idx in enumerate(selected_indices):
        src = Path(paths[idx])
        endonym = endonyms_arr[idx]
        dst_name = f"{rank+1:03d}_{endonym}{src.suffix}"
        shutil.copy2(src, dest / dst_name)

    print(f"Copied {len(selected_indices)} images to: {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description=("Sample curated images by centroid similarity with "
                     "a per-group selection cap"))
    parser.add_argument("--cluster",           required=True)
    parser.add_argument("--manual-dir",        default="data/manual_filtered")
    parser.add_argument("--analysis-dir",      default="data/analysis")
    parser.add_argument("--n",                 type=int,   default=40)
    parser.add_argument("--max-per-endonym",   type=int,   default=8,
                        help="Maximum selected images per endonym group")
    parser.add_argument("--centroid-method",   default="balanced",
                        choices=["global", "balanced"])
    parser.add_argument("--threshold-method",  default="std",
                        choices=["std"],
                        help="Keep images at or above mean - multiplier*std")
    parser.add_argument("--std-multiplier",    type=float, default=1.0,
                        help="Standard-deviation multiplier for the similarity threshold")
    parser.add_argument("--copy",              action="store_true",
                        help="Copy selected images into a new directory under --analysis-dir")
    args = parser.parse_args()

    embeddings_path = Path(args.analysis_dir) / f"{args.cluster}_curated_embeddings.npz"
    if not embeddings_path.exists():
        print(f"Curated embeddings not found: {embeddings_path}")
        return

    embeddings, stored_paths = load_embeddings(embeddings_path)
    paths = [
        str(resolve_saved_path(path).resolve())
        for path in stored_paths
    ]
    images = collect_kept_images(args.manual_dir, args.cluster)
    path_to_endonym = {str(p): e for p, e in images}
    endonyms = [path_to_endonym.get(p, "unknown") for p in paths]

    print(f"\nCluster          : {args.cluster}")
    print(f"Curated images   : {len(embeddings)}")
    print(f"Target N         : {args.n}")
    print(f"Per-group cap    : {args.max_per_endonym} "
          f"({args.max_per_endonym/args.n*100:.0f}% of target N)")
    print(f"Centroid method  : {args.centroid_method}")
    print(f"Threshold method : std (multiplier={args.std_multiplier})")

    # breakdown by stored endonym label
    unique, counts = np.unique(endonyms, return_counts=True)
    print(f"\n── Curated pool by endonym label ──")
    for e, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {e:<25} n={c}")

    centroid = compute_centroid(np.array(embeddings), endonyms, args.centroid_method)

    # compute similarity distribution
    sims_all = np.array(embeddings) @ centroid
    threshold = compute_threshold(sims_all, args.std_multiplier)

    print(f"\n── Similarity distribution ──")
    print(f"  mean      : {sims_all.mean():.3f}")
    print(f"  std       : {sims_all.std():.3f}")
    print(f"  median    : {float(np.median(sims_all)):.3f}")
    print(f"  threshold : {threshold:.3f}  ({(sims_all >= threshold).sum()} / {len(sims_all)} images above)")

    selected_indices, sims, threshold, n_above = select_central_diverse(
        np.array(embeddings), endonyms, paths,
        args.n, args.max_per_endonym, centroid, threshold
    )

    sel_endonyms = np.array(endonyms)[selected_indices]
    sel_sims     = sims[selected_indices]

    print(f"\n── Selected distribution by endonym label ──")
    for e in sorted(set(sel_endonyms)):
        print(f"  {e:<25} {sum(sel_endonyms == e)}")

    print(f"\n── Centroid similarity of selected ──")
    print(f"  mean : {sel_sims.mean():.3f}")
    print(f"  min  : {sel_sims.min():.3f}")
    print(f"  max  : {sel_sims.max():.3f}")

    print(f"\n── Selected image files ──")
    for i, idx in enumerate(selected_indices):
        print(f"  {i+1:>3}. [{sel_endonyms[i]:<20}] {Path(paths[idx]).name}")

    if args.copy:
        copy_selected(
            paths, selected_indices, endonyms, args.cluster,
            args.analysis_dir, args.n, args.max_per_endonym,
            args.centroid_method, args.threshold_method, args.std_multiplier
        )

    print("\nDone!")


if __name__ == "__main__":
    main()