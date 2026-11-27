"""
VLM evaluation script for cultural attribution (country_classification experiment).

Models: gemini (Gemini 2.5 Flash), gpt41 (GPT-4.1), qwen3vl (Qwen3-VL-32B),
        gemma3 (Gemma3-27B)

Result structure:
  results/{cluster_id}/{model}/{visual_cue}/{visual_country}/{experiment}/{textual_cue_type}/{textual_country}/
      img_001.json

Where:
  visual_cue       = neutral | flag | sign | currency
  visual_country   = none (for neutral) | country name
  textual_cue_type = en | endonym | native | perspective
  textual_country  = none (for en) | country name

Examples:
  # Neutral baseline
  results/pilaf_cl/gpt41/neutral/none/country_classification/en/none/

  # Flag only, English prompt
  results/pilaf_cl/gpt41/flag/uzbekistan/country_classification/en/none/

  # Flag + same country endonym
  results/pilaf_cl/gpt41/flag/uzbekistan/country_classification/endonym/uzbekistan/

  # Flag + different country endonym (clash)
  results/pilaf_cl/gpt41/flag/uzbekistan/country_classification/endonym/india/

Usage:
  # Neutral baseline
  python scripts/evaluate.py --model gpt41 --cluster pilaf_cl --cue neutral

  # Visual cue only
  python scripts/evaluate.py --model gpt41 --cluster pilaf_cl --cue flag

  # Visual + same country textual cue
  python scripts/evaluate.py --model gpt41 --cluster pilaf_cl --cue flag --cue-type endonym

  # Clash: uzbekistan flag + india endonym
  python scripts/evaluate.py --model gpt41 --cluster pilaf_cl --cue flag --condition uzbekistan --cue-type endonym --clash-country india
"""

import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

EXPERIMENTS = {
    "country_classification": ["food_name", "primary_country",
                               "confidence_score_for_food",
                               "confidence_score_for_country", "reasoning"],
}

CUE_VALIDATION_FIELDS = {
    "flag": ["primary_country", "confidence_score", "reasoning"],
    "currency": ["primary_country", "confidence_score", "reasoning"],
    "sign": ["sign_text", "language_name", "english_translation", "primary_country"],
}

TEXTUAL_VALIDATION_FIELDS = {
    "endonym": ["english_name", "dish_description"],
    "native": ["language_name", "english_translation", "primary_country"],
}

NO_NEUTRAL_CUES = {"flag", "sign", "currency"}


# ─────────────────────────────────────────────
# PROGRESS TRACKER
# ─────────────────────────────────────────────

class ProgressTracker:
    def __init__(self, total_runs, description=""):
        self.total_runs = total_runs
        self.completed  = 0
        self.start_time = time.time()
        self.description = description
        print(f"\n{'▓'*65}")
        print(f"  STARTING: {description}")
        print(f"  Total images: {total_runs}")
        print(f"  Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'▓'*65}")

    def update(self, n=1):
        self.completed += n
        elapsed  = time.time() - self.start_time
        rate     = self.completed / elapsed if elapsed > 0 else 0
        eta_secs = (self.total_runs - self.completed) / rate if rate > 0 else 0
        eta      = str(timedelta(seconds=int(eta_secs)))
        pct      = self.completed / self.total_runs * 100
        filled   = int(40 * self.completed / self.total_runs)
        bar      = "█" * filled + "░" * (40 - filled)
        print(f"\r  [{bar}] {pct:.1f}% | {self.completed}/{self.total_runs} | ETA: {eta}  ",
              end="", flush=True)

    def done(self):
        elapsed = time.time() - self.start_time
        print(f"\n  Finished in {str(timedelta(seconds=int(elapsed)))}")
        print(f"{'▓'*65}\n")


# ─────────────────────────────────────────────
# PROMPT LOADING
# ─────────────────────────────────────────────

def load_prompt(experiment, prompts_dir="prompts", language="en"):
    """Load baseline prompt from file."""
    is_validation = experiment.endswith("_validation")
    prompt_path = (Path(prompts_dir) / f"{experiment}.txt" if is_validation
                   else Path(prompts_dir) / f"{experiment}_{language}.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def load_prompt_from_cluster(cluster, country_code, cue_type):
    """Load textual cue prompt from clusters.json."""
    key = f"{cue_type}_prompt"
    for country in cluster["countries"]:
        if country["country_code"].lower() == country_code.lower():
            cue_texts = country.get("cue_texts", {})
            if key not in cue_texts:
                raise ValueError(f"No '{key}' for {country_code} in '{cluster['cluster_id']}'.")
            return cue_texts[key]
    raise ValueError(f"Country '{country_code}' not found in '{cluster['cluster_id']}'.")


def get_country_code_for_condition(cluster, condition):
    """Map condition string e.g. 'uzbekistan' to country_code e.g. 'UZ'."""
    for country in cluster["countries"]:
        if country["country"].lower().replace(" ", "_") == condition.lower():
            return country["country_code"]
    raise ValueError(f"Condition '{condition}' not found in cluster '{cluster['cluster_id']}'.")


def load_clusters(path="notes/clusters.json"):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# PATH BUILDER
# ─────────────────────────────────────────────

def build_result_path(results_dir, cluster_id, model, visual_cue, visual_country,
                      experiment, textual_cue_type, textual_country):
    """
    Build result directory path following the unified structure:
    results/{cluster_id}/{model}/{visual_cue}/{visual_country}/{experiment}/{textual_cue_type}/{textual_country}/
    """
    return (Path(results_dir) / cluster_id / model /
            visual_cue / visual_country / experiment /
            textual_cue_type / textual_country)


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def log_header(model, visual_cue, visual_country, textual_cue_type,
               textual_country, experiment):
    clash = (visual_country != textual_country and
             visual_country not in ("none",) and
             textual_country not in ("none",))
    clash_str = " ⚡ CLASH" if clash else ""
    print(f"\n{'█'*65}")
    print(f"  MODEL: {model}{clash_str}")
    print(f"  Visual : {visual_cue} / {visual_country}")
    print(f"  Textual: {textual_cue_type} / {textual_country}")
    print(f"  Experiment: {experiment}")
    print(f"{'█'*65}")


# ─────────────────────────────────────────────
# MODEL BACKENDS
# ─────────────────────────────────────────────

class GeminiBackend:
    model_name = "gemini"
    def __init__(self, api_key_file="notes/gemini_api_key.txt"):
        from google import genai
        with open(api_key_file) as f:
            api_key = f.read().strip()
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"
        print("Gemini backend ready.")
    def query(self, image_path, prompt):
        from PIL import Image
        from google.genai import types
        img = Image.open(image_path).convert("RGB")
        config = types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        return self.client.models.generate_content(
            model=self.model_id, contents=[prompt, img], config=config
        ).text

class Qwen3VLBackend:
    model_name = "qwen3vl"
    def __init__(self, **kwargs):
        import torch
        import os
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        MODEL = os.environ.get(
            "QWEN3VL_MODEL_PATH",
            "path"
        )
        print(f"Loading Qwen3-VL-32B from {MODEL}...")
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        print("Qwen3-VL-32B loaded.")

    def query(self, image_path, prompt):
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        img = Image.open(str(image_path)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)


class GPTBackend:
    model_name = "gpt41"
    def __init__(self, api_key_file="notes/openai_api_key.txt", **kwargs):
        from openai import OpenAI
        with open(api_key_file) as f:
            api_key = f.read().strip()
        self.client = OpenAI(api_key=api_key)
        self.model_id = "gpt-4.1-2025-04-14"
        print("GPT-4.1 backend ready.")

    def query(self, image_path, prompt):
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"
                }}
            ]}],
            temperature=0.0,
            max_completion_tokens=1024
        )
        return response.choices[0].message.content


class Gemma3Backend:
    model_name = "gemma3"
    def __init__(self, model_path=None, **kwargs):
        import torch
        import os
        from transformers import Gemma3ForConditionalGeneration, AutoProcessor
        # allow override via env var or argument
        MODEL = model_path or os.environ.get(
            "GEMMA3_MODEL_PATH",
            "path"
        )
        print(f"Loading Gemma3-27B from {MODEL}...")
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        print("Gemma3-27B loaded.")

    def query(self, image_path, prompt):
        import torch
        from PIL import Image
        img = Image.open(str(image_path)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}
        ]}]
        # Step 1: get text template only
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Step 2: process text + image together
        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)


def load_backend(model_name, device="cuda:1", api_key_file="notes/gemini_api_key.txt"):
    backends = {
        "gemini":   GeminiBackend,
        "gpt41":    GPTBackend,
        "qwen3vl":  Qwen3VLBackend,
        "gemma3":   Gemma3Backend,
    }
    if model_name not in backends:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Supported: {list(backends.keys())}")
    if model_name == "gemini":
        return GeminiBackend(api_key_file)
    if model_name == "gpt41":
        return GPTBackend(api_key_file="notes/openai_api_key.txt")
    return backends[model_name]()


# ─────────────────────────────────────────────
# PARSING + SAVING
# ─────────────────────────────────────────────

def parse_response(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group()
    try:
        return {"success": True, "response": json.loads(text)}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}"}


def save_result(record, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def print_summary(records):
    from collections import Counter
    ok = [r for r in records if r.get("success")]
    print(f"\n  {'─'*50}")
    print(f"  Results: {len(ok)}/{len(records)} successful")
    if ok and "primary_country" in ok[0]:
        for c, n in Counter(r.get("primary_country", "?") for r in ok).most_common():
            print(f"    {c}: {n} ({n/len(ok)*100:.0f}%)")


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_images(backend, image_dir, results_dir, meta,
                    prompt, fields, image_filter=None,
                    overwrite=False, delay=0, tracker=None):
    image_dir   = Path(image_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        print(f"  [skip] {image_dir} not found")
        return []

    images = (
        [p for p in [image_dir / image_filter] if p.exists()] if image_filter
        else sorted(f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    )

    if not images:
        print(f"  [skip] No images in {image_dir}")
        return []

    log_header(meta["model"], meta["visual_cue"], meta["visual_country"],
               meta["textual_cue_type"], meta["textual_country"], meta["experiment"])
    print(f"  {len(images)} images → {results_dir}")

    records = []
    for i, image_path in enumerate(images, 1):
        result_path = results_dir / f"{image_path.stem}.json"
        if result_path.exists() and not overwrite:
            print(f"    [{i:03d}/{len(images)}] [skip] {image_path.name}")
            with open(result_path) as f:
                records.append(json.load(f))
            if tracker:
                tracker.update()
            continue

        print(f"    [{i:03d}/{len(images)}] {image_path.name} ...", end=" ", flush=True)
        try:
            raw    = backend.query(image_path, prompt)
            parsed = parse_response(raw)
        except Exception as e:
            parsed = {"success": False, "error": str(e)}
            raw    = ""

        record = {**meta,
                  "image":   image_path.name,
                  "model":   backend.model_name,
                  "success": parsed["success"],
                  "raw":     raw if isinstance(raw, str) else "",
                  "error":   parsed.get("error", "")}

        if parsed["success"]:
            for field in fields:
                record[field] = parsed["response"].get(field, "")
            print(f"→ {record.get('primary_country','?')} "
                  f"(food:{record.get('confidence_score_for_food','?')} "
                  f"country:{record.get('confidence_score_for_country','?')})")
        else:
            print(f"→ ERROR: {record['error']}")

        save_result(record, result_path)
        records.append(record)
        if tracker:
            tracker.update()
        if delay > 0:
            time.sleep(delay)

    return records

def get_country_cluster_ids(all_clusters, country_slug):
    """Return cluster IDs that contain the given country."""
    return [
        cluster["cluster_id"]
        for cluster in all_clusters
        if any(
            country["country"].lower().replace(" ", "_") == country_slug
            for country in cluster["countries"]
        )
    ]


def run_cue_validation(backend, cue_validation_dir, results_root, cluster,
                       all_clusters, cue_name, processed_countries,
                       prompts_dir="prompts", overwrite=False):
    """Validate canonical visual cues and save results to every matching cluster."""
    prompt = load_prompt(f"{cue_name}_validation", prompts_dir)
    cluster_id = cluster["cluster_id"]
    image_dir = Path(cue_validation_dir) / "visual" / cue_name

    if not image_dir.exists():
        print(f"  No cue validation images at {image_dir}")
        return []

    print(f"\n{'█'*65}")
    print(f"  CUE VALIDATION | MODEL: {backend.model_name} | CUE: {cue_name}")
    print(f"{'█'*65}")

    records = []
    for country_info in cluster["countries"]:
        country_slug = country_info["country"].lower().replace(" ", "_")
        if country_slug in processed_countries:
            continue
        processed_countries.add(country_slug)

        image_path = image_dir / f"{country_slug}.jpg"
        if not image_path.exists():
            print(f"  [{country_slug}] No validation image at {image_path} — skipping")
            continue

        target_cluster_ids = get_country_cluster_ids(all_clusters, country_slug)
        target_paths = [
            Path(results_root) / target_cluster_id / backend.model_name /
            "cue_validation" / cue_name / f"{country_slug}.json"
            for target_cluster_id in target_cluster_ids
        ]
        existing_path = next((path for path in target_paths if path.exists()), None)
        if existing_path and not overwrite:
            print(f"  [skip] {cue_name}/{country_slug}")
            with open(existing_path, encoding="utf-8") as f:
                existing_record = json.load(f)
            for target_cluster_id, target_path in zip(target_cluster_ids, target_paths):
                if not target_path.exists():
                    target_record = {**existing_record, "cluster_id": target_cluster_id}
                    save_result(target_record, target_path)
            current_record = {**existing_record, "cluster_id": cluster_id}
            records.append(current_record)
            continue

        print(f"  [{country_slug}] Validating {cue_name} ...", end=" ", flush=True)
        try:
            raw    = backend.query(image_path, prompt)
            parsed = parse_response(raw)
        except Exception as e:
            parsed = {"success": False, "error": str(e)}
            raw    = ""

        record = {"country": country_slug, "cue": cue_name,
                  "model": backend.model_name, "cluster_id": cluster_id,
                  "success": parsed["success"], "raw": raw,
                  "error": parsed.get("error", "")}
        fields = CUE_VALIDATION_FIELDS[cue_name]
        if parsed["success"]:
            for field in fields:
                record[field] = parsed["response"].get(field, "")
            if cue_name == "sign":
                print(f"→ {record.get('primary_country', '?')}")
            else:
                print(
                    f"→ {record.get('primary_country', '?')} "
                    f"({record.get('confidence_score', '?')})"
                )
        else:
            print(f"→ ERROR: {record['error']}")

        for target_cluster_id, target_path in zip(target_cluster_ids, target_paths):
            target_record = {**record, "cluster_id": target_cluster_id}
            save_result(target_record, target_path)
        records.append({**record, "cluster_id": cluster_id})

    return records


def run_textual_validation(backend, cluster, all_clusters, validation_type,
                            results_root, processed_native_countries,
                            prompts_dir="prompts", overwrite=False):
    """
    Validate textual cues (endonym, native) per country per cluster.

    - endonym/native: text-only query (no image), uses a dummy white image
    - native results are canonical and saved to every matching cluster
    """
    from PIL import Image
    import tempfile

    prompt_file = f"{validation_type}_validation.txt"
    prompt_path = Path(prompts_dir) / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    base_prompt = prompt_path.read_text(encoding="utf-8").strip()

    cluster_id = cluster["cluster_id"]
    print(f"\n{'█'*65}")
    print(f"  TEXTUAL VALIDATION | {backend.model_name} | {validation_type} | {cluster_id}")
    print(f"{'█'*65}")

    # create a small white dummy image for text-only queries
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.new("RGB", (64, 64), (255, 255, 255)).save(tmp.name)
    dummy_path = tmp.name

    records = []
    for country_info in cluster["countries"]:
        country_slug = country_info["country"].lower().replace(" ", "_")
        if validation_type == "native":
            if country_slug in processed_native_countries:
                continue
            processed_native_countries.add(country_slug)

        country_code = country_info["country_code"]
        cue_texts    = country_info.get("cue_texts", {})

        result_path = (Path(results_root) / cluster_id / backend.model_name /
                       "textual_validation" / validation_type / f"{country_slug}.json")
        if validation_type == "native":
            target_cluster_ids = get_country_cluster_ids(all_clusters, country_slug)
            target_paths = [
                Path(results_root) / target_cluster_id / backend.model_name /
                "textual_validation" / validation_type / f"{country_slug}.json"
                for target_cluster_id in target_cluster_ids
            ]
        else:
            target_cluster_ids = [cluster_id]
            target_paths = [result_path]

        existing_path = next((path for path in target_paths if path.exists()), None)
        if existing_path and not overwrite:
            print(f"  [skip] {country_slug}")
            with open(existing_path, encoding="utf-8") as f:
                existing_record = json.load(f)
            for target_cluster_id, target_path in zip(target_cluster_ids, target_paths):
                if not target_path.exists():
                    target_record = {**existing_record, "cluster_id": target_cluster_id}
                    save_result(target_record, target_path)
            records.append({**existing_record, "cluster_id": cluster_id})
            continue

        if validation_type == "endonym":
            text = country_info.get("dish_endonym")
            if not text:
                print(f"  [{country_slug}] No endonym — skipping")
                continue
            prompt   = f"{base_prompt}\n\nDish name: {text}"

        elif validation_type == "native":
            text = cue_texts.get("native_prompt")
            if not text:
                print(f"  [{country_slug}] No native_prompt — skipping")
                continue
            # strip everything from the first { onwards — keep only the question part
            question_only = text.split("\n{")[0].strip()
            prompt = f"{base_prompt}\n\n---TEXT SAMPLE BEGIN---\n{question_only}\n---TEXT SAMPLE END---"

        else:
            raise ValueError(f"Unknown validation_type: {validation_type}")

        print(f"  [{country_slug}] '{text}' ...", end=" ", flush=True)
        try:
            raw    = backend.query(dummy_path, prompt)
            parsed = parse_response(raw)
        except Exception as e:
            parsed = {"success": False, "error": str(e)}
            raw    = ""

        fields = TEXTUAL_VALIDATION_FIELDS[validation_type]

        record = {
            "country":        country_slug,
            "country_code":   country_code,
            "validation_type": validation_type,
            "input_text":     text,
            "cluster_id":     cluster_id,
            "model":          backend.model_name,
            "success":        parsed["success"],
            "raw":            raw,
            "error":          parsed.get("error", ""),
        }
        if parsed["success"]:
            for field in fields:
                record[field] = parsed["response"].get(field, "")

            if validation_type == "endonym":
                display_value = record.get("english_name", "?")
            else:
                display_value = record.get("primary_country", "?")

            print(f"→ {display_value}")
        else:
            print(f"→ ERROR: {record['error']}")

        for target_cluster_id, target_path in zip(target_cluster_ids, target_paths):
            target_record = {**record, "cluster_id": target_cluster_id}
            save_result(target_record, target_path)
        records.append({**record, "cluster_id": cluster_id})

    Path(dummy_path).unlink(missing_ok=True)
    return records

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified VLM evaluation")
    parser.add_argument("--clusters-file",  default="notes/clusters.json")
    parser.add_argument("--cluster",        default=None)
    parser.add_argument("--model",          required=True,
                        choices=["gemini", "gpt41", "qwen3vl", "gemma3"])
    parser.add_argument("--cue",            default=None,
                        help="Visual cue type: original|neutral|flag|sign|currency")
    parser.add_argument("--condition",      default=None,
                        help="Visual cue country (e.g. uzbekistan). If omitted runs all countries.")
    parser.add_argument("--cue-type",       default=None,
                        choices=["endonym","native","perspective"],
                        help="Textual cue type. If omitted uses English baseline prompt.")
    parser.add_argument("--clash-country",  default=None,
                        help="Country whose textual prompt to use. If omitted uses same as --condition.")
    parser.add_argument("--experiment",     default="country_classification",
                        choices=["country_classification"])
    parser.add_argument("--language",       default="en",
                        help="Language for file-based baseline prompt.")
    parser.add_argument("--cue-validation", default=None)
    parser.add_argument("--textual-validation", default=None,
                        choices=["endonym", "native"],
                        help="Run textual cue validation")
    parser.add_argument("--prompts-dir",    default="prompts")
    parser.add_argument("--sampled-dir",    default="data/sampled")
    parser.add_argument("--conditions-dir", default="data/cue_result/tablecloth")
    parser.add_argument("--cue-result-dir", default="data/cue_result")
    parser.add_argument("--cue-validation-dir", default="data/cue_validation")
    parser.add_argument("--results-dir",    default="results")
    parser.add_argument("--image",          default=None)
    parser.add_argument("--device",         default="cuda:1")
    parser.add_argument("--api-key-file",   default="notes/gemini_api_key.txt")
    parser.add_argument("--delay",          type=float, default=0.0)
    parser.add_argument("--overwrite",      action="store_true")
    args = parser.parse_args()

    backend      = load_backend(args.model, args.device, args.api_key_file)
    all_clusters = load_clusters(args.clusters_file)["clusters"]
    clusters     = all_clusters
    if args.cluster:
        clusters = [c for c in clusters if c["cluster_id"] == args.cluster]
        if not clusters:
            print(f"Cluster '{args.cluster}' not found.")
            return

    processed_cue_countries = set()
    processed_native_countries = set()

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        print(f"\n{'='*65}\nCluster: {cluster['cluster_name']}\n{'='*65}")

        # ── Cue validation mode ──────────────────────────────────────
        if args.cue_validation:
            run_cue_validation(
                backend=backend,
                cue_validation_dir=args.cue_validation_dir,
                results_root=args.results_dir,
                cluster=cluster,
                all_clusters=all_clusters,
                cue_name=args.cue_validation,
                processed_countries=processed_cue_countries,
                prompts_dir=args.prompts_dir,
                overwrite=args.overwrite,
            )
            continue
        if args.textual_validation:
            run_textual_validation(
                backend=backend,
                cluster=cluster,
                all_clusters=all_clusters,
                validation_type=args.textual_validation,
                results_root=args.results_dir,
                processed_native_countries=processed_native_countries,
                prompts_dir=args.prompts_dir,
                overwrite=args.overwrite,
            )
            continue

        if not args.cue:
            print("Please specify --cue")
            return

        fields      = EXPERIMENTS[args.experiment]
        base_prompt = load_prompt(args.experiment, args.prompts_dir, args.language)
        all_records = []

        # ── Build visual conditions ──────────────────────────────────
        if args.cue == "original":
            visual_conditions = [("original", "none")]
        elif args.cue == "neutral":
            visual_conditions = [("neutral", "none")]
        else:
            if args.condition:
                visual_conditions = [(args.cue, args.condition)]
            else:
                visual_conditions = [
                    (args.cue, c["country"].lower().replace(" ", "_"))
                    for c in cluster["countries"]
                ]

        # ── Build textual conditions ─────────────────────────────────
        if args.cue_type:
            if args.clash_country:
                # specific clash country
                textual_conditions = [(args.cue_type, args.clash_country)]
            else:
                # all countries for this cue type
                textual_conditions = [
                    (args.cue_type, c["country"].lower().replace(" ", "_"))
                    for c in cluster["countries"]
                ]
        else:
            # English baseline, no textual country
            textual_conditions = [("en", "none")]

        # ── Calculate total for progress tracker ─────────────────────
        total = len(visual_conditions) * len(textual_conditions) * 15
        tracker = ProgressTracker(
            total,
            f"{cluster_id} | visual={args.cue} | "
            f"textual={args.cue_type or 'en'} | "
            f"clash={args.clash_country or 'none'}"
        )

        # ── Run all combinations ─────────────────────────────────────
        for visual_cue, visual_country in visual_conditions:
            for textual_cue_type, textual_country in textual_conditions:

                # Determine image directory
                if visual_cue == "original":
                    image_dir = Path(args.sampled_dir) / cluster_id
                elif visual_cue == "neutral":
                    image_dir = Path(args.conditions_dir) / cluster_id / "neutral"
                else:
                    image_dir = (Path(args.cue_result_dir) / visual_cue /
                                 cluster_id / f"{visual_country}_{visual_cue}")

                # Determine prompt
                if textual_cue_type == "en":
                    prompt = base_prompt
                else:
                    country_code = get_country_code_for_condition(cluster, textual_country)
                    prompt = load_prompt_from_cluster(cluster, country_code, textual_cue_type)

                # Build result directory
                res_dir = build_result_path(
                    args.results_dir, cluster_id, args.model,
                    visual_cue, visual_country,
                    args.experiment, textual_cue_type, textual_country
                )

                meta = {
                    "cluster_id":       cluster_id,
                    "model":            args.model,
                    "visual_cue":       visual_cue,
                    "visual_country":   visual_country,
                    "textual_cue_type": textual_cue_type,
                    "textual_country":  textual_country,
                    "experiment":       args.experiment,
                }

                records = evaluate_images(
                    backend=backend,
                    image_dir=image_dir,
                    results_dir=res_dir,
                    meta=meta, prompt=prompt, fields=fields,
                    image_filter=args.image, overwrite=args.overwrite,
                    delay=args.delay, tracker=tracker,
                )
                all_records.extend(records)

        tracker.done()

        if all_records:
            print_summary(all_records)

    print("\n\nDone!")


if __name__ == "__main__":
    main()