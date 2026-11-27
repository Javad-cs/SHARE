#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

# Load submitted evaluation helpers without modifying them.
sys.path.insert(0, str(Path("scripts").resolve()))
import evaluate as ev  # noqa: E402

ROOT = Path.cwd()
MANIFEST = ROOT / "robustness/repeat_stability/sample_manifest.csv"
OUTDIR = ROOT / "robustness/repeat_stability"
RAW_DIR = OUTDIR / "raw_calls"

OPENAI_KEY = Path("notes/openai_api_key.txt")
GEMINI_KEY = Path("notes/gemini_api_key.txt")

SHARE_UPLOAD_DIR = Path("data/share_upload")
PROMPTS_DIR = ROOT / "prompts"
CLUSTERS_FILE = ROOT / "notes/clusters.json"

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_manifest():
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows

def load_clusters():
    data = ev.load_clusters(str(CLUSTERS_FILE))
    clusters = data["clusters"]
    by_id = {c["cluster_id"]: c for c in clusters}
    return clusters, by_id

class GeminiImageFirstBackend:
    model_name = "gemini"

    def __init__(self, api_key_file):
        from google import genai

        with open(api_key_file) as f:
            api_key = f.read().strip()

        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"
        print("Gemini image-first backend ready.")

    def query(self, image_path, prompt):
        from PIL import Image
        from google.genai import types

        img = Image.open(image_path).convert("RGB")

        config = types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            ),
        )

        # Match the historical Gemini request ordering:
        # image first, then text.
        return self.client.models.generate_content(
            model=self.model_id,
            contents=[img, prompt],
            config=config,
        ).text


def get_backend(model):
    if model == "gpt41":
        return ev.GPTBackend(api_key_file=str(OPENAI_KEY))
    elif model == "gemini":
        return GeminiImageFirstBackend(
            api_key_file=str(GEMINI_KEY)
        )
    else:
        raise ValueError(f"Unsupported model for this check: {model}")

def reconstruct_image_path(row):
    import re

    cluster_id = row["cluster_id"]
    visual_cue = row["visual_cue"]
    visual_country = row["visual_country"]
    original_name = row["image"]

    # Final SHARE upload renamed source files to img_NNN.jpg while
    # preserving the numeric image index.
    m = re.match(r"^(\d+)_", original_name)
    if not m:
        raise ValueError(
            f"Cannot extract image index from manifest filename: {original_name}"
        )

    uploaded_name = f"img_{int(m.group(1)):03d}.jpg"

    if visual_cue == "neutral":
        image_path = (
            SHARE_UPLOAD_DIR /
            cluster_id /
            "neutral" /
            uploaded_name
        )
    elif visual_cue in {"flag", "currency", "sign"}:
        image_path = (
            SHARE_UPLOAD_DIR /
            cluster_id /
            visual_cue /
            visual_country /
            uploaded_name
        )
    else:
        raise ValueError(
            f"Unexpected visual cue in stability manifest: {visual_cue}"
        )

    return image_path

def reconstruct_prompt(row, cluster):
    base_prompt = ev.load_prompt(row["experiment"], str(PROMPTS_DIR), "en")
    textual_cue_type = row["textual_cue_type"]
    textual_country = row["textual_country"]

    if textual_cue_type == "en":
        return base_prompt
    else:
        country_code = ev.get_country_code_for_condition(cluster, textual_country)
        return ev.load_prompt_from_cluster(cluster, country_code, textual_cue_type)

def output_path(model, sample_id, repeat_label):
    return RAW_DIR / model / f"{sample_id}_{repeat_label}.json"

def run_one(backend, row, cluster, repeat_label, delay):
    sample_id = row["stability_sample_id"]
    model = row["model"]

    image_path = reconstruct_image_path(row)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompt = reconstruct_prompt(row, cluster)
    fields = ev.EXPERIMENTS[row["experiment"]]

    outpath = output_path(model, sample_id, repeat_label)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if outpath.exists():
        print(f"[skip] {model} {sample_id} repeat {repeat_label} already exists")
        return

    print(f"[run ] {model} {sample_id} repeat {repeat_label} ...", flush=True)

    try:
        raw = backend.query(image_path, prompt)
        parsed = ev.parse_response(raw)
    except Exception as e:
        raw = ""
        parsed = {"success": False, "error": str(e)}

    record = {
        "stability_sample_id": sample_id,
        "repeat_label": repeat_label,
        "model": model,
        "cluster_id": row["cluster_id"],
        "image": row["image"],
        "visual_cue": row["visual_cue"],
        "visual_country": row["visual_country"],
        "textual_cue_type": row["textual_cue_type"],
        "textual_country": row["textual_country"],
        "experiment": row["experiment"],
        "original_primary_country": row["primary_country"],
        "original_source_json": row["source_json"],
        "image_path": str(image_path),
        "prompt_text": prompt,
        "prompt_sha256": sha256_text(prompt),
        "success": parsed["success"],
        "raw": raw if isinstance(raw, str) else "",
        "error": parsed.get("error", ""),
    }

    if parsed["success"]:
        response = parsed["response"]
        for field in fields:
            record[field] = response.get(field, "")
        print(
            f"      -> {record.get('primary_country','?')} "
            f"(food:{record.get('confidence_score_for_food','?')} "
            f"country:{record.get('confidence_score_for_country','?')})",
            flush=True,
        )
    else:
        print(f"      -> ERROR: {record['error']}", flush=True)

    with outpath.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    if delay > 0:
        time.sleep(delay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-model", type=int, default=None,
                    help="For smoke tests, limit number of sampled conditions per model.")
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"Frozen manifest not found: {MANIFEST}")

    if not OPENAI_KEY.exists():
        raise SystemExit(f"OpenAI key file not found: {OPENAI_KEY}")
    if not GEMINI_KEY.exists():
        raise SystemExit(f"Gemini key file not found: {GEMINI_KEY}")

    rows = load_manifest()
    _, clusters_by_id = load_clusters()

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    for model in ["gpt41", "gemini"]:
        model_rows = sorted(by_model[model], key=lambda r: r["stability_sample_id"])
        if args.max_per_model is not None:
            model_rows = model_rows[:args.max_per_model]

        backend = get_backend(model)

        for row in model_rows:
            cluster = clusters_by_id[row["cluster_id"]]
            for repeat_label in ["A", "B"]:
                run_one(backend, row, cluster, repeat_label, args.delay)

    print("\nDone.")

if __name__ == "__main__":
    main()
