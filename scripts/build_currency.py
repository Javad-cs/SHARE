"""
Build currency cue images by placing a country's banknote onto the table
with perspective homography, then compositing with food images.

Currency images are read from:
  data/cue_references/currency/{country_slug}/verified.jpg

Writes:
  data/cue_result/currency/{cluster_id}/{country_slug}_currency/img_001.jpg

Usage:
    python scripts/build_currency.py --cluster pilaf_cl
    python scripts/build_currency.py --cluster pilaf_cl --country uzbekistan
    python scripts/build_currency.py --cluster pilaf_cl --image img_001.png
    python scripts/build_currency.py --country uzbekistan --table-only
"""

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Quad on table surface where banknote will be placed (in 1024x1024 table space)
BILL_CORNERS = np.float32([
    [720, 595],   # top-left
    [1010, 595],  # top-right
    [1010, 711],  # bottom-right
    [720, 711],   # bottom-left
])

# Table surface boundaries
Y_TOP        = 578
Y_BOTTOM     = 924

# Scaling defaults — area-based (consistent with other build scripts)
AREA_RATIO   = 0.40   # food occupies this fraction of table surface area
MAX_W_RATIO  = 0.55   # safety cap: food width never exceeds this fraction of image width
V_OFFSET     = 0.05   # shift center up by this fraction of table height
FOOD_X_RATIO = 0.4    # horizontal center of food


def load_clusters(path="notes/clusters.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_bbox_from_alpha(food_img):
    alpha = np.array(food_img.split()[3])
    rows  = np.any(alpha > 10, axis=1)
    cols  = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, food_img.width, food_img.height
    return (int(np.where(cols)[0][0]), int(np.where(rows)[0][0]),
            int(np.where(cols)[0][-1]), int(np.where(rows)[0][-1]))


def warp_bill(bill_path, table_img):
    """Warp bill onto table image, return result as PIL Image."""
    bill_img = Image.open(bill_path).convert("RGB")
    tw, th   = table_img.size
    bw, bh   = bill_img.size

    src = np.float32([
        [0,    0   ],
        [bw-1, 0   ],
        [bw-1, bh-1],
        [0,    bh-1],
    ])
    H, _        = cv2.findHomography(src, BILL_CORNERS)
    bill_np     = np.array(bill_img)
    warped      = cv2.warpPerspective(bill_np, H, (tw, th))
    warped_mask = cv2.warpPerspective(
        np.ones((bh, bw), dtype=np.uint8) * 255, H, (tw, th)
    )

    result    = np.array(table_img)
    mask_bool = warped_mask > 128
    result[mask_bool] = warped[mask_bool]
    return Image.fromarray(result)


def composite(food_path, bill_path, output_path,
              y_top=Y_TOP, y_bottom=Y_BOTTOM,
              area_ratio=AREA_RATIO, max_w_ratio=MAX_W_RATIO,
              v_offset=V_OFFSET, table_only=False,
              food_x_ratio=FOOD_X_RATIO):

    table_img = Image.open("data/cue_references/table_1024.jpg").convert("RGB")
    result    = warp_bill(bill_path, table_img)
    tw, th    = table_img.size

    if table_only:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, quality=95)
        return

    # Place food on top — area-based scaling
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
        rows_nw  = np.any(nw, axis=1)
        cols_nw  = np.any(nw, axis=0)
        rmin = int(np.where(rows_nw)[0][0])  if rows_nw.any() else 0
        rmax = int(np.where(rows_nw)[0][-1]) if rows_nw.any() else food_img.height
        cmin = int(np.where(cols_nw)[0][0])  if cols_nw.any() else 0
        cmax = int(np.where(cols_nw)[0][-1]) if cols_nw.any() else food_img.width

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


def main():
    parser = argparse.ArgumentParser(description="Build currency cue images")
    parser.add_argument("--clusters-file", default="notes/clusters.json")
    parser.add_argument("--cluster",       default=None)
    parser.add_argument("--country",       default=None)
    parser.add_argument("--currency-refs", default="data/cue_references/currency")
    parser.add_argument("--extracted-dir", default="data/extracted")
    parser.add_argument("--output-dir",    default="data/cue_result/currency")
    parser.add_argument("--image",         default=None)
    parser.add_argument("--overwrite",     action="store_true")
    parser.add_argument("--table-only",    action="store_true",
                        help="Place bill on empty table only, skip food compositing")
    parser.add_argument("--area-ratio",    type=float, default=AREA_RATIO,
                        help="Food occupies this fraction of table surface area (default: 0.40)")
    parser.add_argument("--max-w-ratio",   type=float, default=MAX_W_RATIO,
                        help="Safety cap: max food width as fraction of image width (default: 0.55)")
    parser.add_argument("--v-offset",      type=float, default=V_OFFSET,
                        help="Shift center up by this fraction of table height (default: 0.05)")
    parser.add_argument("--food-x-ratio",  type=float, default=FOOD_X_RATIO,
                        help="Horizontal center of food (0.5=center, 0.4=slightly left)")
    args = parser.parse_args()

    data     = load_clusters(args.clusters_file)
    clusters = data["clusters"]
    if args.cluster:
        clusters = [c for c in clusters if c["cluster_id"] == args.cluster]
        if not clusters:
            print(f"Cluster '{args.cluster}' not found.")
            return

    # --table-only: warp each country's bill onto bare table and save
    if args.table_only:
        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            for country_info in cluster["countries"]:
                country_slug = country_info["country"].lower().replace(" ", "_")
                if args.country and country_slug != args.country:
                    continue
                bill_path = None
                for ext in IMAGE_EXTENSIONS:
                    candidate = Path(args.currency_refs) / country_slug / f"verified{ext}"
                    if candidate.exists():
                        bill_path = candidate
                        break
                if not bill_path:
                    print(f"[{country_slug}] No verified bill found — skipping")
                    continue
                output_path = Path("data/cue_validation/visual/currency") / f"{country_slug}.jpg"
                if output_path.exists() and not args.overwrite:
                    print(f"[{country_slug}] [skip] already exists: {output_path}")
                    continue
                print(f"[{country_slug}] → {output_path}")
                composite(None, bill_path, output_path, table_only=True)
        print("\nDone!")
        return

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        food_dir   = Path(args.extracted_dir) / cluster_id
        print(f"\n{'='*60}\n{cluster_id}\n{'='*60}")

        if not food_dir.exists():
            print(f"  No extracted images at {food_dir}")
            continue

        if args.image:
            stem   = Path(args.image).stem
            images = [f for ext in IMAGE_EXTENSIONS
                      if (f := food_dir / f"{stem}{ext}").exists()]
        else:
            images = sorted([f for f in food_dir.iterdir()
                             if f.suffix.lower() in IMAGE_EXTENSIONS])

        for country_info in cluster["countries"]:
            country_slug = country_info["country"].lower().replace(" ", "_")
            if args.country and country_slug != args.country:
                continue

            bill_path = None
            for ext in IMAGE_EXTENSIONS:
                candidate = Path(args.currency_refs) / country_slug / f"verified{ext}"
                if candidate.exists():
                    bill_path = candidate
                    break
            if not bill_path:
                print(f"  [{country_slug}] No verified bill found — skipping")
                continue

            out_dir = Path(args.output_dir) / cluster_id / f"{country_slug}_currency"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  [{country_slug}] {len(images)} images → {out_dir}")

            for food_path in images:
                output_path = out_dir / f"{food_path.stem}.jpg"
                if output_path.exists() and not args.overwrite:
                    print(f"    [skip] {food_path.name}")
                    continue
                print(f"    {food_path.name} ...", end=" ", flush=True)
                try:
                    composite(
                        food_path, bill_path, output_path,
                        area_ratio=args.area_ratio,
                        max_w_ratio=args.max_w_ratio,
                        v_offset=args.v_offset,
                        food_x_ratio=args.food_x_ratio,
                    )
                    print("done")
                except Exception as e:
                    print(f"ERROR: {e}")

    print("\n\nDone!")


if __name__ == "__main__":
    main()