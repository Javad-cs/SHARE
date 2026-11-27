"""
Flask-based manual image review tool for scraped food images.
Run on server, access from Mac browser via SSH tunnel.

Setup:
    conda activate env
    pip install flask --break-system-packages
    python scripts/manual_filter.py --cluster pilaf_cl

Then open http://localhost:5000 in your browser.

Keys:
    k      = keep
    1-6    = exclusion criteria
    r      = reset current image decision
    left   = go back
    right  = go forward
"""

import csv
import shutil
import argparse
from pathlib import Path
from flask import Flask, send_file, jsonify, request, render_template_string

CATEGORIES = {
    "k": ("keep",              "Keep"),
    "1": ("e1_food_occluded",  "E1: Food occluded / cut off"),
    "2": ("e2_not_served",     "E2: Not final served state"),
    "3": ("e3_watermark_text", "E3: Watermark / text on food"),
    "4": ("e4_ai_generated",   "E4: Suspected AI generated"),
    "5": ("e5_selective_blur", "E5: Part of dish out of focus"),
    "6": ("e6_other",          "E6: Something else"),
}

CSV_FIELDS = ["image_number", "original_endonym", "original_path",
              "decision", "category_folder", "output_path"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Image Review</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #111; color: #eee; font-family: monospace;
      height: 100vh; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 10px; padding: 12px;
    }
    #status { font-size: 12px; color: #666; letter-spacing: 0.05em; }
    #decision-badge {
      font-size: 13px; padding: 4px 12px; border-radius: 3px;
      background: #1e1e1e; border: 1px solid #444; min-width: 200px;
      text-align: center; min-height: 24px;
    }
    #decision-badge.keep   { border-color: #4a9; color: #4a9; }
    #decision-badge.exclude { border-color: #c55; color: #c55; }
    #decision-badge.none   { color: #555; }
    #image-container {
      flex: 1; display: flex; align-items: center;
      justify-content: center; width: 100%; min-height: 0;
    }
    #food-image { max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 4px; }
    #criteria {
      font-size: 11px; color: #555; line-height: 1.8;
      width: 100%; max-width: 960px;
      border-top: 1px solid #222; padding-top: 6px;
    }
    #buttons { display: flex; gap: 7px; flex-wrap: wrap; justify-content: center; }
    button {
      padding: 7px 12px; border: 1px solid #444; border-radius: 4px;
      background: #1e1e1e; color: #eee; font-family: monospace;
      font-size: 12px; cursor: pointer; transition: background 0.1s;
    }
    button:hover { background: #2a2a2a; border-color: #666; }
    button.keep   { border-color: #4a9; color: #4a9; }
    button.keep:hover { background: #1a3a2a; }
    button.exclude { border-color: #c55; color: #c55; }
    button.exclude:hover { background: #2a1a1a; }
    button.nav    { border-color: #669; color: #99b; }
    button.nav:hover { background: #1a1a2a; }
    button.reset  { border-color: #a80; color: #fa0; }
    button.reset:hover { background: #2a1a00; }
    #hint { font-size: 11px; color: #444; }
    #toast {
      position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
      background: #333; color: #eee; padding: 6px 14px; border-radius: 4px;
      font-size: 12px; opacity: 0; transition: opacity 0.3s; pointer-events: none;
    }
    #toast.show { opacity: 1; }
  </style>
</head>
<body>
  <div id="status">Loading...</div>
  <div id="decision-badge" class="none">—</div>

  <div id="image-container">
    <img id="food-image" src="" alt="food image">
  </div>

  <div id="criteria">
    <b>E1</b>: Food mass occluded/cut off by frame (plate partial = ok) &nbsp;|&nbsp;
    <b>E2</b>: Not final served state (raw/mid-cooking) &nbsp;|&nbsp;
    <b>E3</b>: Watermark/text/object covers food &nbsp;|&nbsp;
    <b>E4</b>: Suspected AI generated &nbsp;|&nbsp;
    <b>E5</b>: Part of dish itself out of focus (background blur = ok) &nbsp;|&nbsp;
    <b>E6</b>: Something else
  </div>

  <div id="buttons">
    <button class="nav"    onclick="navigate(-1)">[ ← ] Back</button>
    <button class="nav"    onclick="navigate(1)" >[ → ] Forward</button>
    <button class="reset"  onclick="resetCurrent()">[ r ] Reset</button>
    <button class="keep"   onclick="decide('k')">[k] Keep</button>
    <button class="exclude" onclick="decide('1')">[1] E1</button>
    <button class="exclude" onclick="decide('2')">[2] E2</button>
    <button class="exclude" onclick="decide('3')">[3] E3</button>
    <button class="exclude" onclick="decide('4')">[4] E4</button>
    <button class="exclude" onclick="decide('5')">[5] E5</button>
    <button class="exclude" onclick="decide('6')">[6] E6</button>
  </div>

  <div id="hint">← → navigate &nbsp;|&nbsp; k 1-6 decide &nbsp;|&nbsp; r reset current</div>
  <div id="toast"></div>

  <script>
    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 1200);
    }

    function updateBadge(decision, label) {
      const b = document.getElementById('decision-badge');
      if (!decision) {
        b.textContent = '— not reviewed yet —';
        b.className = 'none';
      } else if (decision === 'k') {
        b.textContent = '✓ ' + label;
        b.className = 'keep';
      } else {
        b.textContent = '✗ ' + label;
        b.className = 'exclude';
      }
    }

    async function loadCurrent() {
      const res = await fetch('/current');
      const data = await res.json();
      if (data.done) {
        document.getElementById('image-container').innerHTML =
          '<div style="color:#4a9;font-size:18px;text-align:center">All images reviewed.</div>';
        document.getElementById('buttons').style.display = 'none';
        document.getElementById('criteria').style.display = 'none';
        document.getElementById('hint').style.display = 'none';
        document.getElementById('status').textContent = 'Done!';
        document.getElementById('decision-badge').style.display = 'none';
        return;
      }
      document.getElementById('status').textContent =
        '[' + data.index + ' / ' + data.total + ']   reviewed: ' + data.reviewed_count;
      document.getElementById('food-image').src =
        '/image?path=' + encodeURIComponent(data.path) + '&t=' + Date.now();
      updateBadge(data.decision, data.decision_label);
    }

    async function decide(key) {
      await fetch('/decide', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ key })
      });
      await loadCurrent();
      // auto advance to next if not at end
      const res = await fetch('/advance', { method: 'POST' });
      const data = await res.json();
      if (data.advanced) await loadCurrent();
    }

    async function navigate(dir) {
      await fetch('/navigate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ dir })
      });
      await loadCurrent();
    }

    async function resetCurrent() {
      const res = await fetch('/reset', { method: 'POST' });
      const data = await res.json();
      showToast(data.message || 'Reset');
      await loadCurrent();
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft')  { navigate(-1); return; }
      if (e.key === 'ArrowRight') { navigate(1);  return; }
      if (e.key === 'r') { resetCurrent(); return; }
      if (['k','1','2','3','4','5','6'].includes(e.key)) decide(e.key);
    });

    loadCurrent();
  </script>
</body>
</html>
"""

app = Flask(__name__)
state = {}


def collect_images(filtered_dir, cluster_id):
    cluster_path = filtered_dir / cluster_id
    if not cluster_path.exists():
        raise FileNotFoundError(f"No filtered images at {cluster_path}")
    images = []
    for term_dir in sorted(cluster_path.iterdir()):
        if not term_dir.is_dir():
            continue
        endonym = term_dir.name
        for f in sorted(term_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((f.resolve(), endonym))
    return images


def load_log(log_path):
    rows = {}
    if not log_path.exists():
        return rows
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["original_path"]] = row
    return rows


def rewrite_log(log_path, rows):
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)


def save_decision(img_path, endonym, image_number, key):
    category_folder, _ = CATEGORIES[key]
    out_dir = state["output_dir"] / state["cluster_id"] / category_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{image_number:04d}_{endonym}{img_path.suffix.lower()}"
    out_path = out_dir / out_filename

    # remove old output file if exists
    old_row = state["log_rows"].get(str(img_path))
    if old_row:
        old_out = Path(old_row["output_path"])
        if old_out.exists():
            old_out.unlink()

    shutil.copy2(img_path, out_path)

    row = {
        "image_number":     image_number,
        "original_endonym": endonym,
        "original_path":    str(img_path),
        "decision":         key,
        "category_folder":  category_folder,
        "output_path":      str(out_path),
    }
    state["log_rows"][str(img_path)] = row
    rewrite_log(state["log_path"], state["log_rows"])
    return row


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/image")
def serve_image():
    path = request.args.get("path")
    if not path:
        return "no path", 400
    p = Path(path)
    if not p.exists():
        return f"not found: {path}", 404
    ext = p.suffix.lower()
    mimes = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}
    return send_file(str(p.resolve()), mimetype=mimes.get(ext, "image/jpeg"))


@app.route("/current")
def current():
    images = state["images"]
    idx = state["index"]
    if idx >= len(images):
        return jsonify({"done": True})
    img_path, endonym = images[idx]
    existing = state["log_rows"].get(str(img_path))
    decision = existing["decision"] if existing else None
    decision_label = CATEGORIES[decision][1] if decision else None
    reviewed_count = len(state["log_rows"])
    return jsonify({
        "path":          str(img_path),
        "index":         idx + 1,
        "total":         len(images),
        "reviewed_count": reviewed_count,
        "decision":      decision,
        "decision_label": decision_label,
    })


@app.route("/decide", methods=["POST"])
def decide():
    data = request.get_json()
    key = data.get("key")
    if key not in CATEGORIES:
        return jsonify({"error": "invalid key"}), 400
    idx = state["index"]
    images = state["images"]
    if idx >= len(images):
        return jsonify({"done": True})
    img_path, endonym = images[idx]
    image_number = idx + 1
    save_decision(img_path, endonym, image_number, key)
    return jsonify({"ok": True})


@app.route("/advance", methods=["POST"])
def advance():
    """Auto advance to next unreviewed image after a decision."""
    images = state["images"]
    idx = state["index"]
    next_idx = idx + 1
    if next_idx < len(images):
        state["index"] = next_idx
        return jsonify({"advanced": True})
    return jsonify({"advanced": False})


@app.route("/navigate", methods=["POST"])
def navigate():
    data = request.get_json()
    direction = data.get("dir", 0)
    images = state["images"]
    new_idx = state["index"] + direction
    new_idx = max(0, min(new_idx, len(images) - 1))
    state["index"] = new_idx
    return jsonify({"ok": True, "index": new_idx})


@app.route("/reset", methods=["POST"])
def reset_current():
    idx = state["index"]
    images = state["images"]
    if idx >= len(images):
        return jsonify({"message": "No image to reset"})
    img_path, _ = images[idx]
    existing = state["log_rows"].pop(str(img_path), None)
    if existing:
        out_path = Path(existing["output_path"])
        if out_path.exists():
            out_path.unlink()
        rewrite_log(state["log_path"], state["log_rows"])
        return jsonify({"message": "Reset — pick a new decision"})
    return jsonify({"message": "No decision to reset"})


def main():
    parser = argparse.ArgumentParser(description="Manual image review tool")
    parser.add_argument("--cluster",      required=True)
    parser.add_argument("--filtered-dir", default="data/filtered")
    parser.add_argument("--output-dir",   default="data/manual_filtered")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--port",         type=int, default=5000)
    args = parser.parse_args()

    filtered_dir = Path(args.filtered_dir)
    output_dir   = Path(args.output_dir)
    log_path     = output_dir / args.cluster / "filter_log.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = collect_images(filtered_dir, args.cluster)

    if args.resume:
        log_rows = load_log(log_path)
    else:
        if log_path.exists():
            print(f"\n[warn] Log already exists at {log_path}")
            print("  Use --resume to continue, or delete the log to start fresh.")
            return
        log_rows = {}

    # find first unreviewed image to start at
    reviewed_paths = set(log_rows.keys())
    start_idx = 0
    for i, (p, _) in enumerate(all_images):
        if str(p) not in reviewed_paths:
            start_idx = i
            break

    print(f"\nCluster  : {args.cluster}")
    print(f"Total    : {len(all_images)}")
    print(f"Reviewed : {len(log_rows)}")
    print(f"Starting at image {start_idx + 1}")
    print(f"\nOpen http://localhost:{args.port} in your browser")
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port}")

    state["cluster_id"] = args.cluster
    state["images"]     = all_images
    state["index"]      = start_idx
    state["output_dir"] = output_dir
    state["log_path"]   = log_path
    state["log_rows"]   = log_rows

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()