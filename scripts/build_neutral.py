"""
Build neutral condition images by compositing normalized food onto a plain
table surface.

Places food by aligning plate bounding box center with table surface center,
scaling food to occupy a fixed fraction of table surface area (area_ratio),
with a safety cap on maximum width.

Reads:  data/extracted/{cluster_id}/img_001.png  (RGBA, transparent background)
Writes: data/cue_result/tablecloth/{cluster_id}/neutral/img_001.jpg

Usage:
    python scripts/build_neutral.py --cluster pilaf_cl
    python scripts/build_neutral.py --cluster pilaf_cl --image img_001.png
    python scripts/build_neutral.py  # all clusters
"""

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Table surface boundaries in 1024x1024 image
Y_TOP    = 578
Y_BOTTOM = 924

# Scaling defaults
AREA_RATIO   = 0.4    # food occupies this fraction of table surface area
MAX_W_RATIO  = 0.55   # safety cap: food width never exceeds this fraction of image width
V_OFFSET     = 0.05   # shift center up by this fraction of table height
FOOD_X_RATIO = 0.4    # horizontal center of food as fraction of image width (0.5 = centered)


def load_clusters(path="notes/clusters.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_bbox_from_alpha(food_img):
    """Get bounding box of non-transparent content using alpha channel."""
    alpha = np.array(food_img.split()[3])
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, food_img.width, food_img.height
    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1])
    return cmin, rmin, cmax, rmax


def get_bbox_from_threshold(food_img, white_threshold=240):
    """Get bounding box of non-white content for RGB images."""
    arr = np.array(food_img.convert("RGB"))
    is_nonwhite = ~np.all(arr > white_threshold, axis=2)
    rows = np.any(is_nonwhite, axis=1)
    cols = np.any(is_nonwhite, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, food_img.width, food_img.height
    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1])
    return cmin, rmin, cmax, rmax


def composite_food_on_surface(surface_path, food_path, output_path,
                               y_top=Y_TOP, y_bottom=Y_BOTTOM,
                               area_ratio=AREA_RATIO,
                               max_w_ratio=MAX_W_RATIO,
                               v_offset=V_OFFSET,
                               food_x_ratio=FOOD_X_RATIO):
    surface_img = Image.open(surface_path).convert("RGB")
    food_raw    = Image.open(food_path)
    tw, th = surface_img.size

    # Table surface dimensions
    table_h  = y_bottom - y_top
    table_cx = int(tw * food_x_ratio)
    table_cy = (y_top + y_bottom) // 2 - int(table_h * v_offset)

    # Get plate bounding box
    is_rgba = food_raw.mode == "RGBA"
    if is_rgba:
        food_img = food_raw.convert("RGBA")
        cmin, rmin, cmax, rmax = get_bbox_from_alpha(food_img)
    else:
        food_img = food_raw.convert("RGB")
        cmin, rmin, cmax, rmax = get_bbox_from_threshold(food_img)

    bbox_w = cmax - cmin
    bbox_h = rmax - rmin

    if bbox_w == 0 or bbox_h == 0:
        print(f"    WARNING: empty bbox, skipping")
        return

    # Area-based scaling: normalize food to fixed fraction of table surface area
    table_area  = table_h * tw
    target_area = table_area * area_ratio
    food_area   = bbox_w * bbox_h
    scale_area  = (target_area / food_area) ** 0.5

    # Safety cap: food width never exceeds max_w_ratio of image width
    scale_max_w = (tw * max_w_ratio) / bbox_w
    scale = min(scale_area, scale_max_w)

    print(f"    constraint: {'width_cap' if scale_max_w < scale_area else 'area'} "
          f"(scale_area={scale_area:.2f}, scale_max_w={scale_max_w:.2f})")

    new_w    = int(food_img.width  * scale)
    new_h    = int(food_img.height * scale)
    food_img = food_img.resize((new_w, new_h), Image.LANCZOS)

    # Scaled bbox center
    bbox_cx = int((cmin + cmax) / 2 * scale)
    bbox_cy = int((rmin + rmax) / 2 * scale)

    # Paste position: align bbox center with table center
    paste_x = table_cx - bbox_cx
    paste_y = table_cy - bbox_cy

    print(f"    scale {scale:.2f} → paste ({paste_x},{paste_y})")

    # Composite
    result = surface_img.copy()

    if is_rgba:
        result_rgba = result.convert("RGBA")
        result_rgba.alpha_composite(food_img, (paste_x, paste_y))
        result = result_rgba.convert("RGB")
    else:
        arr = np.array(food_img)
        is_white = np.all(arr > 240, axis=2)
        mask = Image.fromarray((~is_white).astype(np.uint8) * 255, mode="L")
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        result.paste(food_img, (paste_x, paste_y), mask=mask)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=95)


def process_cluster(cluster_id, food_dir, table_path, output_dir,
                    image_filter=None, overwrite=False,
                    y_top=Y_TOP, y_bottom=Y_BOTTOM,
                    area_ratio=AREA_RATIO, max_w_ratio=MAX_W_RATIO,
                    v_offset=V_OFFSET, food_x_ratio=FOOD_X_RATIO):

    food_dir   = Path(food_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(table_path).exists():
        print(f"  Table surface not found: {table_path} — skipping")
        return

    if image_filter:
        stem   = Path(image_filter).stem
        images = []
        for ext in IMAGE_EXTENSIONS:
            p = food_dir / f"{stem}{ext}"
            if p.exists():
                images = [p]
                break
    else:
        images = sorted([
            f for f in food_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

    if not images:
        print(f"  No food images in {food_dir}")
        return

    print(f"  {len(images)} images → {output_dir}")

    for food_path in images:
        output_path = output_dir / f"{food_path.stem}.jpg"
        if output_path.exists() and not overwrite:
            print(f"    [skip] {food_path.name}")
            continue
        print(f"    {food_path.name} ...", end=" ", flush=True)
        try:
            composite_food_on_surface(
                table_path, food_path, output_path,
                y_top=y_top, y_bottom=y_bottom,
                area_ratio=area_ratio, max_w_ratio=max_w_ratio,
                v_offset=v_offset, food_x_ratio=food_x_ratio,
            )
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Build neutral condition images: food on plain table surface")
    parser.add_argument("--clusters-file",  default="notes/clusters.json")
    parser.add_argument("--cluster",        default=None,
                        help="Single cluster ID to process (default: all)")
    parser.add_argument("--extracted-dir",  default="data/extracted")
    parser.add_argument("--table",          default="data/cue_references/table_1024.jpg",
                        help="Plain table surface image (1024x1024)")
    parser.add_argument("--output-dir",     default="data/cue_result/tablecloth")
    parser.add_argument("--image",          default=None,
                        help="Process a single image only (e.g. img_001.png)")
    parser.add_argument("--y-top",          type=int,   default=Y_TOP)
    parser.add_argument("--y-bottom",       type=int,   default=Y_BOTTOM)
    parser.add_argument("--area-ratio",     type=float, default=AREA_RATIO,
                        help="Food occupies this fraction of table surface area (default: 0.4)")
    parser.add_argument("--max-w-ratio",    type=float, default=MAX_W_RATIO,
                        help="Max food width as fraction of image width (default: 0.55)")
    parser.add_argument("--v-offset",       type=float, default=V_OFFSET,
                        help="Shift center up by this fraction of table height (default: 0.05)")
    parser.add_argument("--food-x-ratio",   type=float, default=FOOD_X_RATIO,
                        help="Horizontal center of food (0.5=center, 0.4=slightly left)")
    parser.add_argument("--overwrite",      action="store_true",
                        help="Overwrite existing output images")
    args = parser.parse_args()

    data     = load_clusters(args.clusters_file)
    clusters = data["clusters"]
    if args.cluster:
        clusters = [c for c in clusters if c["cluster_id"] == args.cluster]
        if not clusters:
            print(f"Cluster '{args.cluster}' not found.")
            return

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        print(f"\n{'='*60}")
        print(f"Cluster: {cluster['cluster_name']}")
        print(f"{'='*60}")

        food_dir = Path(args.extracted_dir) / cluster_id
        if not food_dir.exists():
            print(f"  No extracted images at {food_dir}")
            continue

        process_cluster(
            cluster_id   = cluster_id,
            food_dir     = food_dir,
            table_path   = args.table,
            output_dir   = Path(args.output_dir) / cluster_id / "neutral",
            image_filter = args.image,
            overwrite    = args.overwrite,
            y_top        = args.y_top,
            y_bottom     = args.y_bottom,
            area_ratio   = args.area_ratio,
            max_w_ratio  = args.max_w_ratio,
            v_offset     = args.v_offset,
            food_x_ratio = args.food_x_ratio,
        )

    print("\n\nDone!")


if __name__ == "__main__":
    main()