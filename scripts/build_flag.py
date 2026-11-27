"""
Build flag cue images by warping country flags onto a toothpick flag template
using perspective homography, then compositing with food images.

Pipeline:
  1. build:     Warp country flag onto template, apply transparency mask → RGBA PNG
  2. composite: Place flag on table, paste food on top

Reads:
  data/cue_references/flag_template.png               — original template
  data/cue_references/flag_template_transparent.png   — SAM-extracted transparent template
  data/cue_references/flag/{country}/verified.png     — country flag images
  data/extracted/{cluster_id}/img_001.png             — extracted food (RGBA)

Writes:
  data/cue_result/flag/{cluster_id}/{country}_flag.png   — transparent flag with country colors
  data/cue_result/flag/{cluster_id}/{country}/img_001.jpg — final composites

Usage:
    python scripts/build_flag.py --cluster pilaf_cl
    python scripts/build_flag.py --cluster pilaf_cl --step build
    python scripts/build_flag.py --cluster pilaf_cl --step composite
    python scripts/build_flag.py --cluster pilaf_cl --country uzbekistan
    python scripts/build_flag.py --cluster pilaf_cl --image img_001.png
"""

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# 4 corners of white flag area in template (1024x1024)
FLAG_CORNERS = np.float32([
    [330, 195],  # top-left
    [754, 243],  # top-right
    [740, 514],  # bottom-right
    [329, 466],  # bottom-left
])

# Table surface boundaries
Y_TOP        = 578
Y_BOTTOM     = 924

# Scaling defaults — area-based (consistent with build_tablecloth.py)
AREA_RATIO   = 0.40   # food occupies this fraction of table surface area
MAX_W_RATIO  = 0.55   # safety cap: food width never exceeds this fraction of image width
V_OFFSET     = 0.05   # shift center up by this fraction of table height
FOOD_X_RATIO = 0.4    # horizontal center of food

# Flag placement on table
FLAG_X_RATIO = 0.85
FLAG_Y_RATIO = 0.62
FLAG_SCALE   = 0.35


def load_clusters(path="notes/clusters.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_flag_image(flag_dir):
    for name in ["verified.png", "verified.jpg", "flag.png", "flag.jpg"]:
        p = Path(flag_dir) / name
        if p.exists():
            return p
    for ext in IMAGE_EXTENSIONS:
        candidates = list(Path(flag_dir).glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


def warp_flag_onto_template(template_path, transparent_template_path,
                             flag_image_path, output_path):
    template = np.array(Image.open(template_path).convert("RGB"))
    flag_img = np.array(Image.open(flag_image_path).convert("RGB"))
    fh, fw   = flag_img.shape[:2]

    src = np.float32([[0, 0], [fw-1, 0], [fw-1, fh-1], [0, fh-1]])
    H, _        = cv2.findHomography(src, FLAG_CORNERS)
    th, tw      = template.shape[:2]
    warped_flag = cv2.warpPerspective(flag_img, H, (tw, th))

    mask     = cv2.warpPerspective(
        np.ones((fh, fw), dtype=np.uint8) * 255, H, (tw, th)
    )
    kernel   = np.ones((3, 3), np.uint8)
    mask     = cv2.erode(mask, kernel, iterations=1)
    mask_3ch = mask[:, :, np.newaxis] / 255.0

    result     = (warped_flag * mask_3ch + template * (1 - mask_3ch)).astype(np.uint8)
    result_img = Image.fromarray(result).convert("RGBA")

    transparent = Image.open(transparent_template_path).convert("RGBA")
    result_img.putalpha(transparent.split()[3])

    output_path = Path(str(output_path).replace(".jpg", ".png"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    print(f"  Saved: {output_path}")
    return output_path


def get_bbox_from_alpha(food_img):
    alpha = np.array(food_img.split()[3])
    rows  = np.any(alpha > 10, axis=1)
    cols  = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, food_img.width, food_img.height
    return (int(np.where(cols)[0][0]), int(np.where(rows)[0][0]),
            int(np.where(cols)[0][-1]), int(np.where(rows)[0][-1]))


def composite_food_with_flag(food_path, flag_path, output_path,
                              y_top=Y_TOP, y_bottom=Y_BOTTOM,
                              area_ratio=AREA_RATIO,
                              max_w_ratio=MAX_W_RATIO,
                              v_offset=V_OFFSET,
                              flag_x_ratio=FLAG_X_RATIO,
                              flag_y_ratio=FLAG_Y_RATIO,
                              flag_scale=FLAG_SCALE,
                              food_x_ratio=FOOD_X_RATIO):
    table_img = Image.open("data/cue_references/table_1024.jpg").convert("RGB")
    flag_img  = Image.open(flag_path).convert("RGBA")
    tw, th    = table_img.size

    # Scale and place flag
    flag_w   = int(tw * flag_scale)
    flag_h   = int(flag_img.height * flag_w / flag_img.width)
    flag_img = flag_img.resize((flag_w, flag_h), Image.LANCZOS)
    flag_x   = int(tw * flag_x_ratio) - flag_w // 2
    flag_y   = int(th * flag_y_ratio) - flag_h // 2

    result = table_img.copy().convert("RGBA")
    result.alpha_composite(flag_img, (flag_x, flag_y))
    result = result.convert("RGB")

    if food_path is None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, quality=95)
        return

    # Paste food on top — area-based scaling
    food_raw = Image.open(food_path)
    table_h  = y_bottom - y_top
    table_cx = int(tw * food_x_ratio)
    table_cy = (y_top + y_bottom) // 2 - int(table_h * v_offset)

    is_rgba = food_raw.mode == "RGBA"
    if is_rgba:
        food_img = food_raw.convert("RGBA")
        cmin, rmin, cmax, rmax = get_bbox_from_alpha(food_img)
    else:
        food_img = food_raw.convert("RGB")
        arr      = np.array(food_img)
        nw       = ~np.all(arr > 240, axis=2)
        rows     = np.any(nw, axis=1)
        cols     = np.any(nw, axis=0)
        rmin = int(np.where(rows)[0][0])  if rows.any() else 0
        rmax = int(np.where(rows)[0][-1]) if rows.any() else food_img.height
        cmin = int(np.where(cols)[0][0])  if cols.any() else 0
        cmax = int(np.where(cols)[0][-1]) if cols.any() else food_img.width

    bbox_w = cmax - cmin
    bbox_h = rmax - rmin
    if bbox_w == 0 or bbox_h == 0:
        print("    WARNING: empty bbox")
        return

    # Area-based scaling
    table_area  = table_h * tw
    target_area = table_area * area_ratio
    food_area   = bbox_w * bbox_h
    scale_area  = (target_area / food_area) ** 0.5
    scale_max_w = (tw * max_w_ratio) / bbox_w
    scale       = min(scale_area, scale_max_w)

    food_img = food_img.resize(
        (int(food_img.width * scale), int(food_img.height * scale)), Image.LANCZOS
    )
    paste_x = table_cx - int((cmin + cmax) / 2 * scale)
    paste_y = table_cy - int((rmin + rmax) / 2 * scale)

    if is_rgba:
        result_rgba = result.convert("RGBA")
        result_rgba.alpha_composite(food_img, (paste_x, paste_y))
        result = result_rgba.convert("RGB")
    else:
        arr  = np.array(food_img)
        mask = Image.fromarray(
            (~np.all(arr > 240, axis=2)).astype(np.uint8) * 255, mode="L"
        ).filter(ImageFilter.GaussianBlur(radius=2))
        result.paste(food_img, (paste_x, paste_y), mask=mask)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=95)


def build_flags(cluster, cue_ref_dir, output_dir,
                template_path, transparent_template_path,
                country_filter=None):
    cluster_id = cluster["cluster_id"]
    print(f"\n{'='*60}\nBuild flags: {cluster_id}\n{'='*60}")

    for country_info in cluster["countries"]:
        country_slug = country_info["country"].lower().replace(" ", "_")
        if country_filter and country_slug != country_filter:
            continue
        flag_dir = Path(cue_ref_dir) / "flag" / country_slug
        flag_img = find_flag_image(flag_dir)
        if not flag_img:
            print(f"  [{country_slug}] No flag image — skipping")
            continue
        output_path = Path(output_dir) / cluster_id / f"{country_slug}_flag.png"
        print(f"  [{country_slug}] {flag_img.name} →", end=" ", flush=True)
        warp_flag_onto_template(
            template_path, transparent_template_path,
            flag_img, output_path
        )


def build_composites(cluster, flag_dir, extracted_dir, output_dir,
                     image_filter=None, overwrite=False, country_filter=None,
                     area_ratio=AREA_RATIO, max_w_ratio=MAX_W_RATIO,
                     v_offset=V_OFFSET, food_x_ratio=FOOD_X_RATIO):
    cluster_id = cluster["cluster_id"]
    food_dir   = Path(extracted_dir) / cluster_id
    print(f"\n{'='*60}\nComposites: {cluster_id}\n{'='*60}")

    if not food_dir.exists():
        print(f"  No extracted images at {food_dir}")
        return

    if image_filter:
        stem   = Path(image_filter).stem
        images = next(
            ([food_dir / f"{stem}{ext}"] for ext in IMAGE_EXTENSIONS
             if (food_dir / f"{stem}{ext}").exists()), []
        )
    else:
        images = sorted([
            f for f in food_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

    for country_info in cluster["countries"]:
        country_slug = country_info["country"].lower().replace(" ", "_")
        if country_filter and country_slug != country_filter:
            continue

        flag_path = Path(flag_dir) / cluster_id / f"{country_slug}_flag.png"
        if not flag_path.exists():
            print(f"  [{country_slug}] No flag found — skipping")
            continue

        out_dir = Path(output_dir) / cluster_id / f"{country_slug}_flag"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [{country_slug}] {len(images)} images → {out_dir}")

        for food_path in images:
            output_path = out_dir / f"{food_path.stem}.jpg"
            if output_path.exists() and not overwrite:
                print(f"    [skip] {food_path.name}")
                continue
            print(f"    {food_path.name} ...", end=" ", flush=True)
            try:
                composite_food_with_flag(
                    food_path, flag_path, output_path,
                    area_ratio=area_ratio, max_w_ratio=max_w_ratio,
                    v_offset=v_offset, food_x_ratio=food_x_ratio,
                )
                print("done")
            except Exception as e:
                print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build flag cue images")
    parser.add_argument("--clusters-file",        default="notes/clusters.json")
    parser.add_argument("--cluster",              default=None)
    parser.add_argument("--country",              default=None)
    parser.add_argument("--step",                 default="both",
                        choices=["build", "composite", "both"])
    parser.add_argument("--template",             default="data/cue_references/flag_template.png")
    parser.add_argument("--transparent-template", default="data/cue_references/flag_template_transparent.png")
    parser.add_argument("--cue-ref-dir",          default="data/cue_references")
    parser.add_argument("--flag-dir",             default="data/cue_result/flag")
    parser.add_argument("--table-only",           action="store_true",
                        help="Place flag on empty table only, skip food compositing")
    parser.add_argument("--extracted-dir",        default="data/extracted")
    parser.add_argument("--output-dir",           default="data/cue_result/flag")
    parser.add_argument("--image",                default=None)
    parser.add_argument("--area-ratio",           type=float, default=AREA_RATIO,
                        help="Food occupies this fraction of table surface area (default: 0.40)")
    parser.add_argument("--max-w-ratio",          type=float, default=MAX_W_RATIO,
                        help="Safety cap: max food width as fraction of image width (default: 0.55)")
    parser.add_argument("--v-offset",             type=float, default=V_OFFSET,
                        help="Shift center up by this fraction of table height (default: 0.05)")
    parser.add_argument("--food-x-ratio",         type=float, default=FOOD_X_RATIO,
                        help="Horizontal center of food (0.5=center, 0.4=slightly left)")
    parser.add_argument("--overwrite",            action="store_true")
    args = parser.parse_args()

    data     = load_clusters(args.clusters_file)
    clusters = data["clusters"]
    if args.cluster:
        clusters = [c for c in clusters if c["cluster_id"] == args.cluster]
        if not clusters:
            print(f"Cluster '{args.cluster}' not found.")
            return

    for p, name in [(args.template, "template"),
                    (args.transparent_template, "transparent template")]:
        if not Path(p).exists():
            print(f"{name} not found: {p}")
            return

    for cluster in clusters:
        if args.step in ("build", "both"):
            build_flags(
                cluster=cluster,
                cue_ref_dir=args.cue_ref_dir,
                output_dir=args.flag_dir,
                template_path=args.template,
                transparent_template_path=args.transparent_template,
                country_filter=args.country,
            )

        if args.table_only:
            cluster_id = cluster["cluster_id"]
            for country_info in cluster["countries"]:
                country_slug = country_info["country"].lower().replace(" ", "_")
                if args.country and country_slug != args.country:
                    continue
                flag_path = Path(args.flag_dir) / cluster_id / f"{country_slug}_flag.png"
                if not flag_path.exists():
                    print(f"[{country_slug}] No flag PNG found — run --step build first")
                    continue
                output_path = Path("data/cue_validation/visual/flag") / f"{country_slug}.jpg"
                if output_path.exists() and not args.overwrite:
                    print(f"[{country_slug}] [skip] already exists: {output_path}")
                    continue
                print(f"[{country_slug}] → {output_path}")
                composite_food_with_flag(None, flag_path, output_path)
            continue

        if args.step in ("composite", "both"):
            build_composites(
                cluster=cluster,
                flag_dir=args.flag_dir,
                extracted_dir=args.extracted_dir,
                output_dir=args.output_dir,
                image_filter=args.image,
                overwrite=args.overwrite,
                country_filter=args.country,
                area_ratio=args.area_ratio,
                max_w_ratio=args.max_w_ratio,
                v_offset=args.v_offset,
                food_x_ratio=args.food_x_ratio,
            )

    print("\n\nDone!")


if __name__ == "__main__":
    main()