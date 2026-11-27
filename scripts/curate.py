"""
Curation pipeline: near-duplicate review followed by two embedding-based filters.

The script writes review copies, stage records,
stage pass/fail snapshots, and NPZ embeddings.

Stage 0: CLIP near-duplicate review
         data/curated/{cluster}/stage0/review/       flagged pairs for inspection
         data/curated/{cluster}/stage0/record.json   pairs and manual decisions

Stage 1: endonym group filtering by separation ratio
         data/curated/{cluster}/stage1/pass/
         data/curated/{cluster}/stage1/fail/
         data/curated/{cluster}/stage1/record.json

Stage 2: image-level filtering by standardized Isolation Forest score
         data/curated/{cluster}/stage2/pass/
         data/curated/{cluster}/stage2/fail/
         data/curated/{cluster}/stage2/record.json

Workflow:
    # Step 1 — generate and inspect the Stage 0 near-duplicate review
    python scripts/curate.py --cluster pilaf_cl --audit

    # Step 2 — record confirmed duplicates, then run the complete pipeline
    python scripts/curate.py --cluster pilaf_cl

    # Run the complete pipeline for all configured clusters
    python scripts/curate.py
"""

import argparse
import csv
import json
import shutil
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLIP_DEDUP_THRESHOLD = 0.95  # cosine similarity above which images are near-duplicates
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_saved_path(path_value):
    """Resolve an old absolute or new project-relative stored path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_project_relative(path_value):
    """Return a portable path relative to the repository root."""
    path = resolve_saved_path(path_value).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Path is outside the project root and cannot be stored portably: {path}"
        ) from exc


# ── CLIP loading ──────────────────────────────────────────────────
_clip_model = None
_clip_preprocess = None


def load_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import torch
        import open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        model = model.to(device).eval()
        _clip_model = model
        _clip_preprocess = preprocess
    return _clip_model, _clip_preprocess


def get_embedding(image_path):
    import torch
    from PIL import Image
    model, preprocess = load_clip()
    device = next(model.parameters()).device
    try:
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
            emb /= emb.norm()
        return emb.cpu().numpy().squeeze()
    except Exception:
        return None


def compute_manual_embeddings(cluster_id, manual_dir, analysis_dir):
    """Compute and save CLIP ViT-L-14 embeddings for all manually kept images."""
    import numpy as np
    from PIL import Image

    keep_path = Path(manual_dir) / cluster_id / "keep"
    log_path  = Path(manual_dir) / cluster_id / "filter_log.csv"
    out_path  = Path(analysis_dir) / f"{cluster_id}_manual_embeddings.npz"

    if not keep_path.exists():
        print(f"  No keep folder at {keep_path}")
        return False

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Load model
    load_clip()

    # Collect images in sorted order
    images = sorted([f for f in keep_path.iterdir()
                     if f.suffix.lower() in IMAGE_EXTENSIONS])

    print(f"  Computing embeddings for {len(images)} images...")
    embeddings = []
    paths      = []
    for i, img_path in enumerate(images):
        emb = get_embedding(img_path)
        if emb is not None:
            embeddings.append(emb)
            paths.append(to_project_relative(img_path))
        else:
            print(f"  WARNING: failed to embed {img_path.name}")
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(images)} done")

    import numpy as np
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    paths_arr      = np.array(paths)

    Path(analysis_dir).mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=embeddings_arr, paths=paths_arr)
    print(f"  Saved {len(embeddings)} embeddings to {out_path}")
    return True


# ── helpers ───────────────────────────────────────────────────────
def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["paths"].tolist()


def load_endonyms(manual_dir, cluster_id, paths):
    log_path = Path(manual_dir) / cluster_id / "filter_log.csv"
    path_to_endonym = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["decision"] == "k":
                path_to_endonym[row["output_path"]] = row["original_endonym"]
    endonyms = []
    for p in paths:
        fname = Path(p).name
        matched = None
        for op, e in path_to_endonym.items():
            if Path(op).name == fname:
                matched = e
                break
        endonyms.append(matched or "unknown")
    return endonyms


# ── stage 0: CLIP near-duplicate audit ───────────────────────────
def stage0_clip_dedup_audit(embeddings, paths, cluster_id, curated_dir,
                             threshold=CLIP_DEDUP_THRESHOLD):
    """
    Find near-duplicate pairs using CLIP embeddings.
    Copies each flagged pair side-by-side to stage0/review/ for inspection.
    Saves record.json with all pairs.
    Returns the number of flagged pairs.
    """
    print(f"\n  Stage 0 — CLIP near-duplicate audit (threshold={threshold})")

    n = len(embeddings)
    emb = np.array(embeddings)
    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / norms

    # pairwise similarity
    sims = emb_norm @ emb_norm.T

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if sims[i, j] >= threshold:
                pairs.append({
                    "path_a": to_project_relative(paths[i]),
                    "path_b": to_project_relative(paths[j]),
                    "similarity": float(sims[i, j]),
                })

    pairs.sort(key=lambda x: -x["similarity"])
    print(f"    Found {len(pairs)} near-duplicate pairs")

    review_dir = curated_dir / "stage0" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    # copy pairs side by side for inspection
    for idx, pair in enumerate(pairs):
        pa = resolve_saved_path(pair["path_a"])
        pb = resolve_saved_path(pair["path_b"])
        sim = pair["similarity"]
        # copy as pair_001_a_filename.jpg and pair_001_b_filename.jpg
        shutil.copy2(pa, review_dir / f"pair_{idx+1:03d}_A_{pa.name}")
        shutil.copy2(pb, review_dir / f"pair_{idx+1:03d}_B_{pb.name}")
        print(f"    pair {idx+1:03d}: sim={sim:.4f}  {pa.name}  <->  {pb.name}")

    # save record
    record = {
        "cluster": cluster_id,
        "stage": 0,
        "method": f"CLIP ViT-L-14 cosine similarity >= {threshold}",
        "threshold": threshold,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "instructions": (
            "Inspect images in stage0/review/. "
            "For each pair, decide which to keep. "
            "Add paths to exclude to 'confirmed_duplicates' list below, "
            "then rerun curate.py without --audit to apply."
        ),
        "confirmed_duplicates": []  # fill this in after inspection
    }

    record_path = curated_dir / "stage0" / "record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"    Review images: {review_dir}")
    print(f"    Record: {record_path}")
    print(f"\n    ── Next steps ──")
    print(f"    1. Open {review_dir} and inspect each pair")
    print(f"    2. Add paths to exclude to 'confirmed_duplicates' in {record_path}")
    print(f"    3. Rerun: python scripts/curate.py --cluster {cluster_id}")

    return len(pairs)


def load_confirmed_duplicates(curated_dir):
    """Load user-confirmed duplicate paths from stage0/record.json."""
    record_path = curated_dir / "stage0" / "record.json"
    if not record_path.exists():
        return set()
    with open(record_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    confirmed = set(record.get("confirmed_duplicates", []))
    if confirmed:
        print(f"\n  Stage 0 — applying {len(confirmed)} confirmed duplicate exclusions")
        for p in sorted(confirmed):
            print(f"    exclude: {Path(p).name}")
    return confirmed


# ── stage 1: endonym group filtering ─────────────────
def compute_separation_ratios(embeddings, endonyms):
    endonyms_arr = np.array(endonyms)
    results = {}
    for e in sorted(set(endonyms)):
        mask = endonyms_arr == e
        rest_mask = ~mask
        if mask.sum() < 2 or rest_mask.sum() < 2:
            continue
        group_embs = embeddings[mask]
        rest_embs  = embeddings[rest_mask]
        gc = group_embs.mean(axis=0); gc /= np.linalg.norm(gc)
        rc = rest_embs.mean(axis=0);  rc /= np.linalg.norm(rc)
        g2r = 1 - (group_embs @ rc)
        r2r = 1 - (rest_embs  @ rc)
        results[e] = {
            "n": int(mask.sum()),
            "separation_ratio": float(g2r.mean() / r2r.mean()),
            "centroid_distance": float(1 - gc @ rc),
        }
    return results


def stage1_endonym_exclusion(embeddings, paths, endonyms, cluster_id,
                              curated_dir, confirmed_dups_names=None):
    ratios = compute_separation_ratios(embeddings, endonyms)
    ratio_values = np.array([v["separation_ratio"] for v in ratios.values()])
    mean = float(ratio_values.mean())
    std  = float(ratio_values.std())
    threshold = float(mean + 2 * std)

    excluded_endonyms = {e for e, v in ratios.items()
                         if v["separation_ratio"] > threshold}

    endonyms_arr = np.array(endonyms)
    paths_arr    = np.array(paths)

    pass_mask = np.array([e not in excluded_endonyms for e in endonyms])
    fail_mask = ~pass_mask

    pass_dir = curated_dir / "stage1" / "pass"
    fail_dir = curated_dir / "stage1" / "fail"
    pass_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    for p, keep in zip(paths, pass_mask):
        src = Path(p)
        if src.exists():
            shutil.copy2(src, (pass_dir if keep else fail_dir) / src.name)

    record = {
        "cluster": cluster_id,
        "stage": 1,
        "method": "group separation ratio > mean + 2*std",
        "threshold": threshold,
        "within_cluster_stats": {"mean": mean, "std": std,
                                  "n_endonyms": len(ratios)},
        "excluded_endonyms": list(excluded_endonyms),
        "all_separation_ratios": ratios,
        "counts": {"input": int(len(paths)),
                   "pass": int(pass_mask.sum()),
                   "fail": int(fail_mask.sum())},
    }
    record_path = curated_dir / "stage1" / "record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"\n  Stage 1 — endonym group filtering")
    print(f"    threshold (mean+2std): {threshold:.4f}")
    if excluded_endonyms:
        for e in excluded_endonyms:
            r = ratios[e]
            print(f"    excluded: {e} "
                  f"(ratio={r['separation_ratio']:.3f}, n={r['n']})")
    else:
        print(f"    no endonym groups excluded")
    print(f"    pass: {pass_mask.sum()} | fail: {fail_mask.sum()}")

    return embeddings[pass_mask], paths_arr[pass_mask].tolist(), \
           endonyms_arr[pass_mask].tolist()


# ── stage 2: Isolation Forest score filtering ────────────────────
def stage2_isolation_forest(embeddings, paths, endonyms, cluster_id,
                             curated_dir, confirmed_dups_names=None):
    iso = IsolationForest(contamination="auto", random_state=42)
    iso.fit(embeddings)
    scores   = iso.decision_function(embeddings)
    z_scores = (scores - scores.mean()) / scores.std()
    pass_mask = z_scores > -2.0
    fail_mask = ~pass_mask

    endonyms_arr = np.array(endonyms)
    paths_arr    = np.array(paths)

    pass_dir = curated_dir / "stage2" / "pass"
    fail_dir = curated_dir / "stage2" / "fail"
    pass_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    for p, keep in zip(paths, pass_mask):
        src = Path(p)
        if not src.exists():
            continue
        if confirmed_dups_names and src.name in confirmed_dups_names:
            continue  # skip confirmed duplicates
        shutil.copy2(src, (pass_dir if keep else fail_dir) / src.name)

    fail_endonyms = endonyms_arr[fail_mask].tolist()
    fail_paths    = paths_arr[fail_mask].tolist()
    fail_zscores  = z_scores[fail_mask].tolist()

    fail_by_endonym = {}
    for e, p, z in zip(fail_endonyms, fail_paths, fail_zscores):
        if e not in fail_by_endonym:
            fail_by_endonym[e] = []
        fail_by_endonym[e].append({
            "path": to_project_relative(p),
            "z_score": float(z),
        })

    record = {
        "cluster": cluster_id,
        "stage": 2,
        "method": "standardized Isolation Forest decision score (Liu et al. 2008)",
        "params": {"contamination": "auto", "z_score_threshold": -2.0,
                   "random_state": 42},
        "counts": {"input": int(len(paths)), "pass": int(pass_mask.sum()),
                   "fail": int(fail_mask.sum()),
                   "fail_pct": float(100 * fail_mask.sum() / len(paths))},
        "failed_by_endonym": fail_by_endonym,
    }
    record_path = curated_dir / "stage2" / "record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"\n  Stage 2 — Isolation Forest score filtering")
    print(f"    pass: {pass_mask.sum()} | "
          f"fail: {fail_mask.sum()} "
          f"({100*fail_mask.sum()/len(paths):.1f}%)")
    if fail_by_endonym:
        print(f"    filtered images by endonym label:")
        for e, items in sorted(fail_by_endonym.items(),
                                key=lambda x: -len(x[1])):
            print(f"      {e:<25} {len(items)}")

    return embeddings[pass_mask], paths_arr[pass_mask].tolist(), \
           endonyms_arr[pass_mask].tolist()


# ── main pipeline ─────────────────────────────────────────────────
def curate_cluster(cluster_id, manual_dir, analysis_dir, curated_dir,
                   audit_only=False):
    print(f"\n{'='*55}")
    print(f"Cluster: {cluster_id}")

    emb_path = Path(analysis_dir) / f"{cluster_id}_manual_embeddings.npz"
    if not emb_path.exists():
        print(f"  No embeddings found — computing now...")
        success = compute_manual_embeddings(cluster_id, manual_dir, analysis_dir)
        if not success:
            print(f"  Skipping {cluster_id}")
            return

    embeddings, stored_paths = load_embeddings(emb_path)
    paths = [
        str(resolve_saved_path(path).resolve())
        for path in stored_paths
    ]
    endonyms = load_endonyms(manual_dir, cluster_id, paths)
    cluster_curated = Path(curated_dir) / cluster_id
    cluster_curated.mkdir(parents=True, exist_ok=True)

    print(f"  Input: {len(paths)} images")

    if audit_only:
        # just run stage 0 audit and stop
        stage0_clip_dedup_audit(embeddings, paths, cluster_id, cluster_curated)
        return

    # load confirmed duplicates from stage0 record (if exists)
    confirmed_dups = load_confirmed_duplicates(cluster_curated)

    # apply confirmed duplicate exclusions
    if confirmed_dups:
        # Compare by filename only — full paths may differ across servers
        confirmed_names = {Path(p).name for p in confirmed_dups}
        keep_mask = np.array([Path(p).name not in confirmed_names for p in paths])
        embeddings = embeddings[keep_mask]
        paths      = np.array(paths)[keep_mask].tolist()
        endonyms   = np.array(endonyms)[keep_mask].tolist()
        print(f"  After duplicate exclusion: {len(paths)} images")

    # build set of confirmed duplicate filenames for file copying skip
    confirmed_dups_names = {Path(p).name for p in confirmed_dups}

    # stage 1
    emb1, paths1, end1 = stage1_endonym_exclusion(
        embeddings, paths, endonyms, cluster_id, cluster_curated,
        confirmed_dups_names=confirmed_dups_names)

    # stage 2
    emb2, paths2, end2 = stage2_isolation_forest(
        emb1, paths1, end1, cluster_id, cluster_curated,
        confirmed_dups_names=confirmed_dups_names)

    # save final curated embeddings
    final_emb_path = Path(analysis_dir) / f"{cluster_id}_curated_embeddings.npz"
    relative_paths2 = np.array([to_project_relative(path) for path in paths2])
    np.savez(final_emb_path, embeddings=emb2, paths=relative_paths2)

    print(f"\n  Final retained: {len(paths2)} images")
    print(f"  Curated embeddings: {final_emb_path}")


def main():
    parser = argparse.ArgumentParser(
        description=("Near-duplicate review followed by group- and "
                     "image-level embedding filters"))
    parser.add_argument("--cluster",      default=None,
                        help="Cluster ID; omit to process all configured clusters")
    parser.add_argument("--manual-dir",   default="data/manual_filtered")
    parser.add_argument("--analysis-dir", default="data/analysis")
    parser.add_argument("--curated-dir",  default="data/curated")
    parser.add_argument("--audit",        action="store_true",
                        help="Run Stage 0 only: find near-duplicates and copy "
                             "flagged pairs to review folder for inspection")
    parser.add_argument("--embed",        action="store_true",
                        help="Force recompute manual embeddings even if file exists")
    args = parser.parse_args()

    clusters = (["pilaf_cl", "samosa_cl", "mantou_cl", "meatball_cl"]
                if not args.cluster else [args.cluster])

    if args.audit:
        print("Loading CLIP model for near-duplicate detection...")
        load_clip()
        print("CLIP model loaded.\n")

    for cluster_id in clusters:
        if args.embed:
            print(f"\nComputing manual embeddings for {cluster_id}...")
            compute_manual_embeddings(cluster_id, args.manual_dir, args.analysis_dir)
        curate_cluster(
            cluster_id,
            manual_dir   = args.manual_dir,
            analysis_dir = args.analysis_dir,
            curated_dir  = args.curated_dir,
            audit_only   = args.audit,
        )

    curated_root = Path(args.curated_dir)
    analysis_root = Path(args.analysis_dir)

    if args.audit:
        print("\n\nAudit complete.")
        print(f"Review directory: {curated_root}/{{cluster}}/stage0/review/")
        print("Record confirmed exclusions in:")
        print(f"  {curated_root}/{{cluster}}/stage0/record.json")
        print("Then rerun scripts/curate.py for the same cluster and output directories.")
    else:
        print("\n\nCuration complete.")
        print(f"Stage outputs:      {curated_root}")
        print(f"Embedding outputs:  {analysis_root}")
        next_cluster = args.cluster or "<cluster>"
        print(
            "Next: python scripts/sample.py "
            f"--cluster {next_cluster} "
            f"--manual-dir {args.manual_dir} "
            f"--analysis-dir {args.analysis_dir} --copy"
        )


if __name__ == "__main__":
    main()