"""
Extract clean food+plate mask from white-background Flux images using GDINO + SAM.

Usage:
    # Test on single image
    python scripts/extract_dish.py --image data/normalized_flux/central_asian_plov/img_001.jpg

    # Process full cluster
    python scripts/extract_dish.py --cluster central_asian_plov
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch

# GDINO
from groundingdino.util.inference import load_model as load_gdino, predict
from groundingdino.util import box_ops
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, disk

# SAM
from segment_anything import sam_model_registry, SamPredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

GDINO_CONFIG_PATH = "path"
GDINO_WEIGHTS     = "path"
SAM_CHECKPOINT = "path"
SAM_MODEL_TYPE = "vit_h"
DEVICE = "cuda:1"


def load_models():
    print("Loading GDINO...")
    gdino = load_gdino(GDINO_CONFIG_PATH, GDINO_WEIGHTS)
    gdino = gdino.to(DEVICE)

    print("Loading SAM...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam = sam.to(DEVICE)
    predictor = SamPredictor(sam)

    print("Models loaded.")
    return gdino, predictor


def get_best_box(gdino, image_pil, prompt, box_threshold=0.3, text_threshold=0.25):
    """Run GDINO and return the highest confidence box in xyxy format."""
    import torchvision.transforms as T
    from groundingdino.util.inference import load_image

    # Convert PIL to the format GDINO expects
    transform = T.Compose([
        T.Resize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil.convert("RGB"))

    boxes, logits, phrases = predict(
        model=gdino,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE,
    )

    if len(boxes) == 0:
        return None

    # Convert from cx,cy,w,h normalized to xyxy pixel coords
    h, w = image_pil.height, image_pil.width
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])

    # Take highest confidence box
    best_idx = logits.argmax().item()
    box = boxes_xyxy[best_idx].cpu().numpy().astype(int)
    return box  # [x1, y1, x2, y2]


def get_sam_mask(predictor, image_np, box):
    """Run SAM with box prompt, return best mask."""
    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True,
    )
    return masks[scores.argmax()]  # H x W boolean


def extract_dish_mask(gdino, sam_predictor, image_pil):
    image_np = np.array(image_pil.convert("RGB"))
    h, w = image_np.shape[:2]
    combined_mask = np.zeros((h, w), dtype=bool)

    # Try each prompt separately and combine all masks
    prompts = ["food", "plate", "bowl", "dish"]
    
    for prompt in prompts:
        box = get_best_box(gdino, image_pil, prompt)
        if box is not None:
            mask = get_sam_mask(sam_predictor, image_np, box)
            combined_mask |= mask
            print(f"    '{prompt}' → mask added")

    combined_mask = binary_fill_holes(combined_mask)  # fill internal holes
    combined_mask = closing(combined_mask, disk(5))    # close small gaps between masks
    combined_mask = binary_fill_holes(combined_mask)  # fill again after closing

    # Fallback if nothing found
    if not combined_mask.any():
        print(f"    Nothing found — falling back to center point")
        sam_predictor.set_image(image_np)
        masks, scores, _ = sam_predictor.predict(
            point_coords=np.array([[w//2, h//2]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        combined_mask = masks[scores.argmax()]

    # Convert to RGBA
    rgba = Image.fromarray(image_np).convert("RGBA")
    alpha = Image.fromarray((combined_mask * 255).astype(np.uint8), mode="L")
    rgba.putalpha(alpha)
    return rgba, combined_mask


def get_plate_diameter(mask):
    """Measure plate diameter from mask bounding box."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return max(rmax - rmin, cmax - cmin)


def process_image(gdino, sam_predictor, image_path, output_path):
    image_pil = Image.open(image_path).convert("RGB")
    print(f"    Extracting: {image_path.name} ...", end=" ", flush=True)

    rgba, mask = extract_dish_mask(gdino, sam_predictor, image_pil)
    diameter = get_plate_diameter(mask)
    print(f"plate diameter: {diameter}px")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path)  # save as PNG to preserve alpha
    return diameter


def main():
    parser = argparse.ArgumentParser(description="Extract dish masks using GDINO+SAM")
    parser.add_argument("--cluster",        default=None)
    parser.add_argument("--clusters-file",  default="notes/clusters.json")
    parser.add_argument("--input-dir",      default=None)
    parser.add_argument("--normalized-dir", default="data/normalized_flux")
    parser.add_argument("--output-dir",     default="data/extracted")
    parser.add_argument("--image",          default=None)
    parser.add_argument("--overwrite",      action="store_true")
    args = parser.parse_args()

    gdino, sam_predictor = load_models()

    # Direct folder mode
    if args.input_dir:
        input_dir  = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        images = [input_dir / args.image] if args.image else sorted([
            f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        for img_path in images:
            out_path = output_dir / f"{img_path.stem}.png"
            if out_path.exists() and not args.overwrite:
                print(f"    [skip] {img_path.name}")
                continue
            process_image(gdino, sam_predictor, img_path, out_path)
        return

    # Cluster mode
    import json
    with open(args.clusters_file) as f:
        data = json.load(f)
    clusters = data["clusters"]
    if args.cluster:
        clusters = [c for c in clusters if c["cluster_id"] == args.cluster]

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        print(f"\n{'='*60}\nCluster: {cluster['cluster_name']}\n{'='*60}")

        input_dir  = Path(args.normalized_dir) / cluster_id
        output_dir = Path(args.output_dir) / cluster_id

        if not input_dir.exists():
            print(f"  No images at {input_dir}")
            continue

        if args.image:
            images = [input_dir / args.image]
            images = [p for p in images if p.exists()]
        else:
            images = sorted([
                f for f in input_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ])

        print(f"  {len(images)} images → {output_dir}")
        for img_path in images:
            out_path = output_dir / f"{img_path.stem}.png"
            if out_path.exists() and not args.overwrite:
                print(f"    [skip] {img_path.name}")
                continue
            process_image(gdino, sam_predictor, img_path, out_path)

    print("\nDone!")


if __name__ == "__main__":
    main()