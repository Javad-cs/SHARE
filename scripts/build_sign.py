"""
Build sign cue images by rendering native script text onto a wooden tent sign
template using perspective homography, then compositing with food images.

Sign text is read from clusters.json: country -> cue_texts -> sign

Pipeline:
  1. build:     Render text onto sign face using perspective warp → RGBA PNG
  2. composite: Place sign on table left side, paste food on top

Reads:
  data/cue_references/sign_template.png       — original sign template
  data/cue_references/sign_template_transparent.png   — SAM-extracted transparent template
  data/extracted/{cluster_id}/img_001.png             — extracted food (RGBA)

Writes:
  data/cue_result/sign/{cluster_id}/{country}_sign.png   — sign with native text
  data/cue_result/sign/{cluster_id}/{country}/img_001.jpg — final composites

Usage:
    python scripts/build_sign.py --cluster pilaf_cl
    python scripts/build_sign.py --cluster pilaf_cl --step build
    python scripts/build_sign.py --cluster pilaf_cl --step composite
    python scripts/build_sign.py --cluster pilaf_cl --country uzbekistan
    python scripts/build_sign.py --cluster pilaf_cl --image img_001.png
"""

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Noto Sans — covers Latin + Cyrillic + more
FONT_PATH            = "data/cue_references/fonts/NotoSans-Bold.ttf"
FONT_PATH_ARABIC     = "data/cue_references/fonts/NotoSansArabic-Bold.ttf"
FONT_PATH_CHINESE    = "data/cue_references/fonts/NotoSansSC-Bold.ttf"
FONT_PATH_DEVANAGARI = "data/cue_references/fonts/NotoSansDevanagari-Bold.ttf"
FONT_PATH_JAPANESE   = "data/cue_references/fonts/NotoSansJP-Bold.ttf"
FONT_PATH_KOREAN     = "data/cue_references/fonts/NotoSansKR-Bold.ttf"

# 4 corners of sign face template (1024x1024)
SIGN_CORNERS = np.float32([
    [121, 258],  # top-left
    [906, 258],  # top-right
    [904, 759],  # bottom-right
    [101, 759],  # bottom-left
])

# Table surface boundaries
Y_TOP        = 578
Y_BOTTOM     = 924

# Scaling defaults — area-based (consistent with build_tablecloth.py)
AREA_RATIO   = 0.40   # food occupies this fraction of table surface area
MAX_W_RATIO  = 0.55   # safety cap: food width never exceeds this fraction of image width
V_OFFSET     = 0.05   # shift center up by this fraction of table height
FOOD_X_RATIO = 0.4    # horizontal center of food

# Sign placement
SIGN_X_RATIO = 0.85
SIGN_Y_RATIO = 0.62
SIGN_SCALE   = 0.30


def load_clusters(path="notes/clusters.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_font_for_text(text):
    for char in text:
        if '\u0600' <= char <= '\u06FF':
            return FONT_PATH_ARABIC
        if '\u4e00' <= char <= '\u9fff':
            return FONT_PATH_CHINESE
        if '\u0900' <= char <= '\u097F':
            return FONT_PATH_DEVANAGARI
        if '\u3040' <= char <= '\u30ff':
            return FONT_PATH_JAPANESE
        if '\uAC00' <= char <= '\uD7A3':
            return FONT_PATH_KOREAN
    return FONT_PATH


def get_auto_font_size(text, face_w, face_h, font_path,
                        target_area_ratio=0.15, max_size=300, min_size=20):
    """Scale font so text ink area occupies target_area_ratio of sign face."""
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    target_area = face_w * face_h * target_area_ratio
    
    lo, hi = min_size, max_size
    while lo < hi - 1:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except Exception:
            break
        bbox = dummy.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        text_area = tw * th
        if text_area < target_area:
            lo = mid
        else:
            hi = mid

    # debug
    try:
        font = ImageFont.truetype(font_path, lo)
        bbox = dummy.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        print(f"    final: font_size={lo}, text={tw}x{th}px, area={tw*th}/{int(face_w*face_h*target_area_ratio)}px² ({tw*th/(face_w*face_h)*100:.1f}% of face)")
    except:
        pass
    return lo


def render_text_on_face(text, face_w, face_h, target_area_ratio=0.15):
    font_path = get_font_for_text(text)
    font_size = get_auto_font_size(text, face_w, face_h, font_path,
                                    target_area_ratio=target_area_ratio)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        print(f"    WARNING: font not found at {font_path}, using default")
        font = ImageFont.load_default()

    canvas = Image.new("RGBA", (face_w, face_h), (255, 255, 255, 0))
    draw   = ImageDraw.Draw(canvas)

    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (face_w - tw) // 2 - bbox[0]
    y = (face_h - th) // 2 - bbox[1]

    draw.text((x, y), text, font=font, fill=(60, 30, 10, 255))
    print(f"    font_size={font_size}, text_w={tw}px / face_w={face_w}px")
    return canvas


def warp_text_onto_template(template_path, transparent_template_path,
                             text, output_path):
    template = np.array(Image.open(template_path).convert("RGB"))
    th, tw   = template.shape[:2]

    face_w = int(max(
        np.linalg.norm(SIGN_CORNERS[1] - SIGN_CORNERS[0]),
        np.linalg.norm(SIGN_CORNERS[2] - SIGN_CORNERS[3])
    ))
    face_h = int(max(
        np.linalg.norm(SIGN_CORNERS[3] - SIGN_CORNERS[0]),
        np.linalg.norm(SIGN_CORNERS[2] - SIGN_CORNERS[1])
    ))

    text_canvas = render_text_on_face(text, face_w, face_h)
    text_np     = np.array(text_canvas)

    src = np.float32([
        [0,        0       ],
        [face_w-1, 0       ],
        [face_w-1, face_h-1],
        [0,        face_h-1],
    ])
    H, _         = cv2.findHomography(src, SIGN_CORNERS)
    warped_rgb   = cv2.warpPerspective(text_np[:, :, :3], H, (tw, th))
    warped_alpha = cv2.warpPerspective(text_np[:, :, 3],  H, (tw, th))

    alpha_3ch  = warped_alpha[:, :, np.newaxis] / 255.0
    result     = (warped_rgb * alpha_3ch + template * (1 - alpha_3ch)).astype(np.uint8)
    result_img = Image.fromarray(result).convert("RGBA")

    transparent = Image.open(transparent_template_path).convert("RGBA")
    result_img.putalpha(transparent.split()[3])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    print(f"  Saved: {output_path}")


def get_bbox_from_alpha(food_img):
    alpha = np.array(food_img.split()[3])
    rows  = np.any(alpha > 10, axis=1)
    cols  = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, food_img.width, food_img.height
    return (int(np.where(cols)[0][0]), int(np.where(rows)[0][0]),
            int(np.where(cols)[0][-1]), int(np.where(rows)[0][-1]))


def composite_food_with_sign(food_path, sign_path, output_path,
                              y_top=Y_TOP, y_bottom=Y_BOTTOM,
                              area_ratio=AREA_RATIO,
                              max_w_ratio=MAX_W_RATIO,
                              v_offset=V_OFFSET,
                              sign_x_ratio=SIGN_X_RATIO,
                              sign_y_ratio=SIGN_Y_RATIO,
                              sign_scale=SIGN_SCALE,
                              food_x_ratio=FOOD_X_RATIO):
    table_img = Image.open("data/cue_references/table_1024.jpg").convert("RGB")
    sign_img  = Image.open(sign_path).convert("RGBA")
    tw, th    = table_img.size

    sign_w   = int(tw * sign_scale)
    sign_h   = int(sign_img.height * sign_w / sign_img.width)
    sign_img = sign_img.resize((sign_w, sign_h), Image.LANCZOS)
    sign_x   = int(tw * sign_x_ratio) - sign_w // 2
    sign_y   = int(th * sign_y_ratio) - sign_h // 2

    result = table_img.copy().convert("RGBA")
    result.alpha_composite(sign_img, (sign_x, sign_y))
    result = result.convert("RGB")

    result_np = np.array(result)
    scale_x   = sign_w / 1024
    scale_y   = sign_h / 1024
    scaled_corners = (SIGN_CORNERS * np.array([scale_x, scale_y])).astype(np.int32)
    scaled_corners[:, 0] += sign_x
    scaled_corners[:, 1] += sign_y
    cv2.polylines(result_np, [scaled_corners], isClosed=True,
                  color=(80, 50, 20), thickness=1)
    result = Image.fromarray(result_np)

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


def build_signs(cluster, output_dir, template_path,
                transparent_template_path, country_filter=None):
    cluster_id = cluster["cluster_id"]
    print(f"\n{'='*60}\nBuild signs: {cluster_id}\n{'='*60}")

    for country_info in cluster["countries"]:
        country_slug = country_info["country"].lower().replace(" ", "_")
        if country_filter and country_slug != country_filter:
            continue

        text = country_info.get("cue_texts", {}).get("sign")
        if not text:
            print(f"  [{country_slug}] No sign text in clusters.json — skipping")
            continue

        output_path = Path(output_dir) / cluster_id / f"{country_slug}_sign.png"
        print(f"  [{country_slug}] '{text}' →", end=" ", flush=True)
        warp_text_onto_template(
            template_path, transparent_template_path, text, output_path
        )


def build_composites(cluster, sign_dir, extracted_dir, output_dir,
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

        sign_path = Path(sign_dir) / cluster_id / f"{country_slug}_sign.png"
        if not sign_path.exists():
            print(f"  [{country_slug}] No sign found — skipping")
            continue

        out_dir = Path(output_dir) / cluster_id / f"{country_slug}_sign"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [{country_slug}] {len(images)} images → {out_dir}")

        for food_path in images:
            output_path = out_dir / f"{food_path.stem}.jpg"
            if output_path.exists() and not overwrite:
                print(f"    [skip] {food_path.name}")
                continue
            print(f"    {food_path.name} ...", end=" ", flush=True)
            try:
                composite_food_with_sign(
                    food_path, sign_path, output_path,
                    area_ratio=area_ratio, max_w_ratio=max_w_ratio,
                    v_offset=v_offset, food_x_ratio=food_x_ratio,
                )
                print("done")
            except Exception as e:
                print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build sign cue images")
    parser.add_argument("--clusters-file",        default="notes/clusters.json")
    parser.add_argument("--cluster",              default=None)
    parser.add_argument("--country",              default=None)
    parser.add_argument("--step",                 default="both",
                        choices=["build", "composite", "both"])
    parser.add_argument("--template",             default="data/cue_references/sign_template.png")
    parser.add_argument("--transparent-template", default="data/cue_references/sign_template_transparent.png")
    parser.add_argument("--sign-dir",             default="data/cue_result/sign")
    parser.add_argument("--table-only",           action="store_true",
                        help="Place sign on empty table only, skip food compositing")
    parser.add_argument("--extracted-dir",        default="data/extracted")
    parser.add_argument("--output-dir",           default="data/cue_result/sign")
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
            build_signs(
                cluster=cluster,
                output_dir=args.sign_dir,
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
                sign_path = Path(args.sign_dir) / cluster_id / f"{country_slug}_sign.png"
                if not sign_path.exists():
                    print(f"[{country_slug}] No sign PNG found — run --step build first")
                    continue
                output_path = Path("data/cue_validation/visual/sign") / f"{country_slug}.jpg"
                if output_path.exists() and not args.overwrite:
                    print(f"[{country_slug}] [skip] already exists: {output_path}")
                    continue
                print(f"[{country_slug}] → {output_path}")
                composite_food_with_sign(None, sign_path, output_path)
            continue

        if args.step in ("composite", "both"):
            build_composites(
                cluster=cluster,
                sign_dir=args.sign_dir,
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