"""
Interactive incremental normalization script.

For each image, generates variants using FLUX.1 Kontext [dev] via ComfyUI,
incrementally trying seeds until the user approves a result. Matches the
normalization procedure described in the paper.

Workflow per image:
  1. Try standard prompt (P1) with seeds 1..max_attempts
  2. If all fail, try angle-omitted prompt (P2) with seeds 1..max_attempts
  3. If all fail, flag image for manual intervention (see paper appendix)

Verification criteria (check each generated image against these):
  - Single dish present
  - Food appearance unchanged from original
  - Clean white background
  - Plain unpatterned plating

Note: 4 images required additional manual intervention beyond this script
(plate smoothness prompt and two-pass approach). See paper appendix for details.

Output:
  data/normalized_multi/{cluster}/{img_stem}/p{prompt_idx}_s{seed_idx}.jpg
  data/normalized_multi/{cluster}_selections.json

Usage:
  python scripts/normalize.py --port 8188 --cluster pilaf_cl
  python scripts/normalize.py --port 8188 --cluster pilaf_cl --max-attempts 7
  python scripts/normalize.py --port 8188 --cluster pilaf_cl --image 001_плов.jpg
"""

import json
import time
import argparse
import uuid
import urllib.request
import urllib.parse
import copy
import subprocess
import sys
from pathlib import Path

COMFYUI_HOST  = "127.0.0.1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Base seed — matches paper appendix
BASE_SEED = 431287824357772

# ── Prompts ───────────────────────────────────────────────────────

# P1: Standard prompt (with viewing angle adjustment)
PROMPT_STANDARD = (
    "• Leave all the food quality the exact same as the original.\n"
    "• Modify the background image to a pure white.\n"
    "• Rotate the food items along the z-axis so they are viewed from a natural "
    "dining perspective - not from the top, but more like how someone sees the "
    "plate or bowl while sitting at a table.\n"
    "• Remove any spoons, chopsticks, and human hands.\n"
    "• Replace the pot if any with the plain white bowl.\n"
    "• Replace the pan if any with the plain white plate.\n"
    "• Add any missing parts of a plate.\n"
    "• The plate or bowl should be circle or oval.\n"
    "• The generated image should have only one food item."
)

# P2: Angle-omitted prompt (applied when P1 fails after max_attempts)
PROMPT_NO_ANGLE = (
    "• Leave all the food quality the exact same as the original.\n"
    "• Modify the background image to a pure white.\n"
    "• Remove any spoons, chopsticks, and human hands.\n"
    "• Replace the pot if any with the plain white bowl.\n"
    "• Replace the pan if any with the plain white plate.\n"
    "• Add any missing parts of a plate.\n"
    "• The plate or bowl should be circle or oval.\n"
    "• The generated image should have only one food item."
)

PROMPTS = [
    ("p1", "standard",  PROMPT_STANDARD),
    ("p2", "no_angle",  PROMPT_NO_ANGLE),
]

# ── ComfyUI workflow ──────────────────────────────────────────────

BASE_WORKFLOW = {
    "136":     {"inputs": {"filename_prefix": "normalized",
                           "images": ["192:8", 0]},
                "class_type": "SaveImage"},
    "190":     {"inputs": {"image": "PLACEHOLDER"},
                "class_type": "LoadImage"},
    "192:39":  {"inputs": {"vae_name": "ae.safetensors"},
                "class_type": "VAELoader"},
    "192:135": {"inputs": {"conditioning": ["192:6", 0]},
                "class_type": "ConditioningZeroOut"},
    "192:8":   {"inputs": {"samples": ["192:31", 0],
                           "vae": ["192:39", 0]},
                "class_type": "VAEDecode"},
    "192:35":  {"inputs": {"guidance": 2.5,
                           "conditioning": ["192:177", 0]},
                "class_type": "FluxGuidance"},
    "192:37":  {"inputs": {"unet_name": "flux1-kontext-dev.safetensors",
                           "weight_dtype": "default"},
                "class_type": "UNETLoader"},
    "192:177": {"inputs": {"conditioning": ["192:6", 0],
                           "latent": ["192:124", 0]},
                "class_type": "ReferenceLatent"},
    "192:146": {"inputs": {"direction": "right", "match_image_size": True,
                           "spacing_width": 0, "spacing_color": "white",
                           "image1": ["190", 0]},
                "class_type": "ImageStitch"},
    "192:42":  {"inputs": {"image": ["192:146", 0]},
                "class_type": "FluxKontextImageScale"},
    "192:6":   {"inputs": {"text": "PLACEHOLDER", "clip": ["192:38", 0]},
                "class_type": "CLIPTextEncode"},
    "192:38":  {"inputs": {"clip_name1": "clip_l.safetensors",
                           "clip_name2": "t5xxl_fp16.safetensors",
                           "type": "flux", "device": "default"},
                "class_type": "DualCLIPLoader"},
    "192:124": {"inputs": {"pixels": ["192:42", 0], "vae": ["192:39", 0]},
                "class_type": "VAEEncode"},
    "192:31":  {"inputs": {"seed": BASE_SEED, "steps": 20, "cfg": 1,
                           "sampler_name": "euler", "scheduler": "simple",
                           "denoise": 1, "model": ["192:37", 0],
                           "positive": ["192:35", 0],
                           "negative": ["192:135", 0],
                           "latent_image": ["192:193", 0]},
                "class_type": "KSampler"},
    "192:193": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                "class_type": "EmptySD3LatentImage"},
}


# ── ComfyUI API ───────────────────────────────────────────────────

def api_get(url, endpoint):
    with urllib.request.urlopen(f"{url}/{endpoint}", timeout=30) as r:
        return json.loads(r.read())


def api_post(url, endpoint, data):
    body = json.dumps(data).encode("utf-8")
    req  = urllib.request.Request(
        f"{url}/{endpoint}", data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def upload_image(url, image_path):
    import mimetypes
    filename  = Path(image_path).name
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    boundary  = uuid.uuid4().hex
    with open(image_path, "rb") as f:
        data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{url}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["name"]


def build_workflow(input_filename, prompt_text, seed):
    wf = copy.deepcopy(BASE_WORKFLOW)
    wf["190"]["inputs"]["image"]      = input_filename
    wf["192:6"]["inputs"]["text"]     = prompt_text
    wf["192:31"]["inputs"]["seed"]    = seed
    return {"client_id": str(uuid.uuid4()), "prompt": wf}


def wait_for_result(url, prompt_id, timeout=300, poll_interval=3):
    start = time.time()
    while time.time() - start < timeout:
        history = api_get(url, f"history/{prompt_id}")
        if prompt_id in history:
            for node_output in history[prompt_id].get("outputs", {}).values():
                if "images" in node_output:
                    return node_output["images"]
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out after {timeout}s")


def download_output(url, img_info, output_path):
    params = urllib.parse.urlencode({
        "filename": img_info["filename"],
        "subfolder": img_info.get("subfolder", ""),
        "type": "output"
    })
    with urllib.request.urlopen(f"{url}/view?{params}", timeout=30) as r:
        data = r.read()
    with open(output_path, "wb") as f:
        f.write(data)


def generate_one(url, uploaded_name, prompt, seed, output_path):
    wf        = build_workflow(uploaded_name, prompt, seed)
    result    = api_post(url, "prompt", wf)
    prompt_id = result["prompt_id"]
    images    = wait_for_result(url, prompt_id)
    download_output(url, images[0], output_path)


# ── Image display ─────────────────────────────────────────────────

def open_image(path):
    """Open image in default viewer for user inspection."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform == "win32":
            subprocess.run(["start", str(path)], shell=True, check=False)
        else:
            # Linux — try common viewers
            for viewer in ["eog", "display", "feh", "xdg-open"]:
                try:
                    subprocess.Popen([viewer, str(path)],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    break
                except FileNotFoundError:
                    continue
    except Exception:
        pass  # if no viewer available, user checks path manually


def ask_user(prompt_text):
    """Ask user to approve or reject. Returns True if approved."""
    print(f"\n  Verification criteria:")
    print(f"    1. Single dish present")
    print(f"    2. Food appearance unchanged from original")
    print(f"    3. Clean white background")
    print(f"    4. Plain unpatterned plating")
    while True:
        ans = input(f"\n  {prompt_text} [y=pass / n=fail / q=quit]: ").strip().lower()
        if ans in ("y", "yes"):
            return "pass"
        if ans in ("n", "no"):
            return "fail"
        if ans in ("q", "quit"):
            return "quit"
        print("  Please enter y, n, or q")


# ── Main processing ───────────────────────────────────────────────

def process_image(url, img_path, img_out_dir, max_attempts, overwrite=False):
    """
    Try prompts and seeds incrementally until user approves.
    Returns (selected_key, status) where status is 'pass' or 'manual'.
    """
    uploaded = upload_image(url, img_path)

    for p_idx, (p_key, p_name, prompt_text) in enumerate(PROMPTS, 1):
        print(f"\n  Prompt: {p_name} ({p_key})")
        for s_idx in range(1, max_attempts + 1):
            seed     = BASE_SEED + (s_idx - 1)
            out_key  = f"{p_key}_s{s_idx}"
            out_path = img_out_dir / f"{out_key}.jpg"

            if out_path.exists() and not overwrite:
                print(f"    [{out_key}] already exists — showing for review")
            else:
                print(f"    [{out_key}] generating (seed={seed})...", end=" ", flush=True)
                try:
                    generate_one(url, uploaded, prompt_text, seed, out_path)
                    print("done")
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            # Show image and ask user
            open_image(out_path)
            print(f"    Image saved to: {out_path}")
            result = ask_user(f"Accept {out_key}?")

            if result == "pass":
                return out_key, "pass"
            elif result == "quit":
                return None, "quit"
            # else: fail → continue to next seed

        print(f"\n  All {max_attempts} attempts with {p_name} failed.")
        if p_idx < len(PROMPTS):
            print(f"  Switching to next prompt variant...")

    # All prompts and seeds exhausted
    print(f"\n  ⚠ All attempts failed — flagging for manual intervention.")
    print(f"  See paper appendix for plate smoothness and two-pass procedures.")
    return "MANUAL", "manual"


def process_cluster(url, cluster, input_dir, output_dir, selections_path,
                    max_attempts, image_filter=None, overwrite=False):

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing selections
    selections = {}
    if selections_path.exists():
        with open(selections_path, "r", encoding="utf-8") as f:
            selections = json.load(f)

    images = sorted(f for f in input_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS)

    if image_filter:
        images = [f for f in images if f.name == image_filter or
                  f.stem == Path(image_filter).stem]

    print(f"\n  {len(images)} images to process")
    print(f"  Max attempts per prompt: {max_attempts}")
    print(f"  Prompts: {len(PROMPTS)} "
          f"({', '.join(p[1] for p in PROMPTS)})")
    print(f"  Base seed: {BASE_SEED}")

    for img_path in images:
        img_stem    = img_path.stem
        img_out_dir = output_dir / img_stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already selected and not overwriting
        if img_stem in selections and not overwrite:
            print(f"\n  [{img_stem}] already selected: "
                  f"{selections[img_stem]} — skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"  Image: {img_stem}")
        print(f"  Open original to compare: {img_path}")
        open_image(img_path)

        selected_key, status = process_image(
            url, img_path, img_out_dir, max_attempts, overwrite)

        if status == "quit":
            print("\n  Quit — saving progress and exiting.")
            break

        if selected_key:
            selections[img_stem] = selected_key if status == "pass" else "MANUAL"
            # Save after each image so progress is never lost
            with open(selections_path, "w", encoding="utf-8") as f:
                json.dump(selections, f, indent=2, ensure_ascii=False)
            print(f"\n  Saved: {img_stem} → {selections[img_stem]}")

    # Summary
    total    = len(images)
    passed   = sum(1 for v in selections.values() if v != "MANUAL")
    manual   = sum(1 for v in selections.values() if v == "MANUAL")
    print(f"\n{'='*60}")
    print(f"  Done. {passed}/{total} passed, {manual} flagged for manual intervention.")
    print(f"  Selections saved to: {selections_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive incremental normalization via ComfyUI")
    parser.add_argument("--port",         type=int, required=True,
                        help="ComfyUI port (e.g. 8188)")
    parser.add_argument("--cluster",      required=True,
                        help="Cluster ID (e.g. pilaf_cl)")
    parser.add_argument("--input-dir",    default=None,
                        help="Override input dir "
                             "(default: data/analysis/{cluster}_sample_n40_max8_balanced_std1.0)")
    parser.add_argument(
        "--output-dir",
        default="data/normalized_multi",
        help=(
            "Output directory for generated Flux variants and selections "
            "(default: data/normalized_multi)"
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=7,
                        help="Maximum seed attempts per prompt (default: 7, "
                             "matching paper)")
    parser.add_argument("--image",        default=None,
                        help="Process a single image only (e.g. 001_плов.jpg)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Regenerate even if output already exists")
    args = parser.parse_args()

    url = f"http://{COMFYUI_HOST}:{args.port}"

    try:
        api_get(url, "system_stats")
        print(f"Connected to ComfyUI at {url}")
    except Exception as e:
        print(f"Cannot connect to ComfyUI at {url}: {e}")
        return

    input_dir = (args.input_dir or
                 f"data/analysis/{args.cluster}_sample_n40_max8_balanced_std1.0")
    output_dir      = Path(args.output_dir) / args.cluster
    selections_path = Path(args.output_dir) / f"{args.cluster}_selections.json"

    print(f"Cluster       : {args.cluster}")
    print(f"Input         : {input_dir}")
    print(f"Output        : {output_dir}")
    print(f"Max attempts  : {args.max_attempts} per prompt")
    print(f"Selections    : {selections_path}")

    process_cluster(
        url           = url,
        cluster       = args.cluster,
        input_dir     = input_dir,
        output_dir    = output_dir,
        selections_path = selections_path,
        max_attempts  = args.max_attempts,
        image_filter  = args.image,
        overwrite     = args.overwrite,
    )


if __name__ == "__main__":
    main()
