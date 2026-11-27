"""
Interactive SAM masking tool — box + click + mask accumulation.

Workflow:
  1. Draw a box over food/plate → SAM instantly generates mask
  2. Click + to add regions, click - to remove regions
  3. Happy with mask? Press K to keep it
  4. Draw another box / click more → keep again
  5. Press P to preview union of all kept masks
  6. Press S to save final result

Controls:
  B          — box mode
  F          — add foreground point
  X          — remove background point
  K          — keep current mask
  P          — preview union of kept masks
  C          — clear current prompts
  Z          — undo last action
  Space      — re-run SAM
  ,          — previous image
  S          — save and next
  N          — skip and next

Usage:
    python scripts/interactive_mask.py \
        --input-dir data/sampled/central_asian_plov \
        --output-dir data/normalized/central_asian_plov
"""

import os
import argparse
import base64
import numpy as np
from pathlib import Path
from io import BytesIO

from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import binary_fill_holes, binary_closing
from flask import Flask, request, jsonify


SAM_WEIGHTS = "path"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)
predictor = None
current_image = None
input_dir = None
output_dir = None


def load_sam(device="cuda:1"):
    global predictor
    print("Loading SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS)
    sam.to(device)
    predictor = SamPredictor(sam)
    print("SAM loaded.")


def image_to_base64(img, format="JPEG"):
    buf = BytesIO()
    if format == "PNG":
        img.save(buf, format="PNG")
    else:
        img.convert("RGB").save(buf, format=format, quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def robust_fill(mask, closing_radius=8):
    struct = np.ones((closing_radius, closing_radius), dtype=bool)
    mask = binary_closing(mask, structure=struct)
    mask = binary_fill_holes(mask)
    return mask


def mask_to_overlay(mask, color=(76, 175, 80), alpha=120):
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask] = [*color, alpha]
    return image_to_base64(Image.fromarray(overlay, "RGBA"), format="PNG")


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SAM Interactive Masker</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: sans-serif; background: #1a1a1a; color: #eee; user-select: none; }
.layout { display: flex; height: 100vh; }
.sidebar { width: 230px; min-width: 230px; background: #252525; border-right: 1px solid #333; padding: 14px; display: flex; flex-direction: column; gap: 8px; overflow-y: auto; }
.main { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: auto; padding: 16px; }
h2 { font-size: 14px; font-weight: 600; color: #ccc; }
.file-list { display: flex; flex-direction: column; gap: 3px; max-height: 150px; overflow-y: auto; }
.file-item { padding: 5px 8px; border-radius: 5px; cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #ccc; }
.file-item:hover { background: #333; }
.file-item.active { background: #2a4a7f; color: white; }
.file-item.done::after { content: " ✓"; color: #4caf50; }
button { padding: 7px 10px; border-radius: 5px; border: 1px solid #444; background: #333; cursor: pointer; font-size: 12px; width: 100%; color: #eee; margin-bottom: 2px; }
button:hover:not(:disabled) { background: #404040; }
button:disabled { opacity: 0.4; cursor: default; }
button.primary { background: #1a5fc8; border-color: #1a5fc8; color: white; }
button.primary:hover:not(:disabled) { background: #1452b0; }
button.keep { background: #1a4a2a; border-color: #4caf50; color: #4caf50; }
button.keep:hover:not(:disabled) { background: #1f5a32; }
button.preview-btn { background: #2a2a4a; border-color: #7986cb; color: #7986cb; }
button.preview-btn:hover:not(:disabled) { background: #32325a; }
.mode-row { display: flex; gap: 5px; }
.mode-row button { flex: 1; font-size: 11px; padding: 6px 4px; }
button.mode-active-box { background: #4a4a1a; border-color: #ffd600; color: #ffd600; }
button.mode-active-fg  { background: #1a4a1a; border-color: #4caf50; color: #4caf50; }
button.mode-active-bg  { background: #4a1a1a; border-color: #f44336; color: #f44336; }
.status { font-size: 11px; color: #888; line-height: 1.5; min-height: 28px; }
.kept-count { font-size: 12px; color: #4caf50; font-weight: 500; padding: 3px 0; }
hr { border: none; border-top: 1px solid #333; }
.canvas-wrap { position: relative; display: inline-block; line-height: 0; }
canvas { display: block; background: transparent; }
#overlayCanvas { position: absolute; top: 0; left: 0; pointer-events: none; }
#uiCanvas { position: absolute; top: 0; left: 0; }
.shortcut { font-size: 11px; color: #555; line-height: 1.9; }
.legend { display: flex; flex-direction: column; gap: 4px; font-size: 11px; color: #888; }
.legend-row { display: flex; align-items: center; gap: 6px; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <div>
      <h2>SAM Masker</h2>
      <div class="status" id="status">Loading...</div>
    </div>
    <hr>
    <div style="font-size:11px;color:#666;margin-bottom:2px">Images</div>
    <div class="file-list" id="fileList"></div>
    <hr>
    <div style="font-size:11px;color:#666;margin-bottom:4px">Mode</div>
    <div class="mode-row">
      <button id="btnBox" onclick="setMode('box')" title="Draw box (B)">□ Box</button>
      <button id="btnFg"  onclick="setMode('fg')"  title="Add to mask (F)">+ Add</button>
      <button id="btnBg"  onclick="setMode('bg')"  title="Remove (X)">- Remove</button>
    </div>
    <hr>
    <div class="legend">
      <div class="legend-row"><div class="dot" style="background:#ffd600"></div> Box — initial region</div>
      <div class="legend-row"><div class="dot" style="background:#4caf50"></div> Green — add to mask</div>
      <div class="legend-row"><div class="dot" style="background:#f44336"></div> Red — remove from mask</div>
    </div>
    <hr>
    <button onclick="undoLast()">Undo (Z)</button>
    <button onclick="clearCurrent()">Clear current (C)</button>
    <hr>
    <button class="keep" id="keepBtn" onclick="keepMask()" disabled>Keep mask (K)</button>
    <div class="kept-count" id="keptCount">0 masks kept</div>
    <button class="preview-btn" id="previewBtn" onclick="previewUnion()" disabled>Preview union (P)</button>
    <button onclick="clearKept()" id="clearKeptBtn" style="font-size:11px;padding:5px">Clear kept masks</button>
    <hr>
    <button class="primary" id="saveBtn" onclick="saveMask()" disabled>Save (S)</button>
    <button onclick="nextImage()">Skip / Next (N)</button>
    <hr>
    <div class="shortcut">
      B — box mode<br>
      F — add foreground<br>
      X — remove background<br>
      K — keep current mask<br>
      P — preview union<br>
      Z — undo last<br>
      C — clear current<br>
      Space — re-run SAM<br>
      S — save &amp; next<br>
      N — skip
    </div>
  </div>

  <div class="main">
    <div class="canvas-wrap" id="wrap">
      <canvas id="imgCanvas"></canvas>
      <canvas id="overlayCanvas"></canvas>
      <canvas id="uiCanvas"></canvas>
    </div>
  </div>
</div>

<script>
let files = [], fileIdx = 0, currentFile = null;
let mode = 'box';
let box = null;
let points = [];
let history = [];
let maskB64 = null;
let keptMasks = [];
let scale = 1;
let origImg = null;
let drawing = false;
let drawStart = null;

const imgCanvas     = document.getElementById('imgCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const uiCanvas      = document.getElementById('uiCanvas');
const ic  = imgCanvas.getContext('2d');
const oc  = overlayCanvas.getContext('2d');
const uc  = uiCanvas.getContext('2d');
const status  = document.getElementById('status');
const saveBtn = document.getElementById('saveBtn');
const keepBtn = document.getElementById('keepBtn');
const previewBtn = document.getElementById('previewBtn');

// ── file list ──────────────────────────────────────────────────────────────
fetch('/files').then(r=>r.json()).then(data => {
  files = data.files;
  const list = document.getElementById('fileList');
  files.forEach((f,i) => {
    const el = document.createElement('div');
    el.className = 'file-item';
    el.textContent = f;
    el.id = 'fi_'+i;
    el.onclick = () => loadFile(i);
    list.appendChild(el);
  });
  if (files.length) loadFile(0);
});

function loadFile(idx) {
  fileIdx = idx; currentFile = files[idx];
  box = null; points = []; history = [];
  maskB64 = null; keptMasks = [];
  saveBtn.disabled = true;
  keepBtn.disabled = true;
  previewBtn.disabled = true;
  document.getElementById('keptCount').textContent = '0 masks kept';
  document.querySelectorAll('.file-item').forEach((el,i) => el.classList.toggle('active', i===idx));
  status.textContent = 'Loading ' + currentFile + '...';
  fetch('/load?file=' + encodeURIComponent(currentFile))
    .then(r=>r.json()).then(data => {
      const img = new Image();
      img.onload = () => {
        origImg = img;
        const maxW = window.innerWidth - 250;
        const maxH = window.innerHeight - 32;
        scale = Math.min(1, maxW/img.width, maxH/img.height);
        const cw = Math.round(img.width * scale);
        const ch = Math.round(img.height * scale);
        [imgCanvas, overlayCanvas, uiCanvas].forEach(c => {
          c.width = cw; c.height = ch;
        });
        ic.drawImage(img, 0, 0, cw, ch);
        oc.clearRect(0,0,cw,ch);
        uc.clearRect(0,0,cw,ch);
        setMode('box');
        status.textContent = currentFile + ' — draw a box over food+plate';
      };
      img.src = 'data:image/jpeg;base64,' + data.image;
    });
}

// ── canvas interactions ────────────────────────────────────────────────────
function canvasPos(e) {
  const r = uiCanvas.getBoundingClientRect();
  return {
    cx: e.clientX - r.left,
    cy: e.clientY - r.top,
    ix: Math.round((e.clientX - r.left) / scale),
    iy: Math.round((e.clientY - r.top) / scale),
  };
}

uiCanvas.addEventListener('mousedown', e => {
  e.preventDefault();
  const pos = canvasPos(e);
  if (mode === 'box') {
    drawing = true;
    drawStart = pos;
    history.push({type:'box', prev: box});
    box = null;
  } else {
    const label = mode === 'fg' ? 1 : 0;
    history.push({type:'point', point: {x:pos.ix, y:pos.iy, label}});
    points.push({x:pos.ix, y:pos.iy, label});
    drawUI();
    runSAM();
  }
});

uiCanvas.addEventListener('mousemove', e => {
  if (!drawing || mode !== 'box') return;
  const pos = canvasPos(e);
  box = {x1:drawStart.ix, y1:drawStart.iy, x2:pos.ix, y2:pos.iy};
  drawUI();
});

uiCanvas.addEventListener('mouseup', e => {
  if (!drawing) return;
  drawing = false;
  const pos = canvasPos(e);
  if (mode === 'box') {
    box = {x1:drawStart.ix, y1:drawStart.iy, x2:pos.ix, y2:pos.iy};
    const bw = Math.abs(box.x2-box.x1), bh = Math.abs(box.y2-box.y1);
    if (bw > 10 && bh > 10) {
      drawUI();
      runSAM();
    } else {
      box = null;
      drawUI();
    }
  }
});

uiCanvas.addEventListener('mouseleave', () => { drawing = false; });
uiCanvas.addEventListener('contextmenu', e => e.preventDefault());

// ── draw UI ────────────────────────────────────────────────────────────────
function drawUI() {
  uc.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
  if (box) {
    const x1 = Math.min(box.x1,box.x2)*scale, y1 = Math.min(box.y1,box.y2)*scale;
    const x2 = Math.max(box.x1,box.x2)*scale, y2 = Math.max(box.y1,box.y2)*scale;
    uc.strokeStyle = '#ffd600';
    uc.lineWidth = 2;
    uc.setLineDash([6,3]);
    uc.strokeRect(x1, y1, x2-x1, y2-y1);
    uc.setLineDash([]);
  }
  points.forEach(p => {
    uc.beginPath();
    uc.arc(p.x*scale, p.y*scale, 7, 0, Math.PI*2);
    uc.fillStyle = p.label===1 ? 'rgba(76,175,80,0.9)' : 'rgba(244,67,54,0.9)';
    uc.fill();
    uc.strokeStyle = 'white';
    uc.lineWidth = 2;
    uc.stroke();
    uc.fillStyle = 'white';
    uc.font = 'bold 10px sans-serif';
    uc.textAlign = 'center';
    uc.textBaseline = 'middle';
    uc.fillText(p.label===1 ? '+' : '−', p.x*scale, p.y*scale);
  });
}

function drawOverlay(previewB64) {
  const tmp = new Image();
  tmp.onload = () => {
    oc.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    oc.drawImage(tmp, 0, 0, overlayCanvas.width, overlayCanvas.height);
    drawUI();
  };
  tmp.src = 'data:image/png;base64,' + previewB64;
}

// ── SAM ────────────────────────────────────────────────────────────────────
function runSAM() {
  if (!box && points.length === 0) return;
  status.textContent = 'Running SAM...';
  const payload = {
    file: currentFile,
    box: box ? [
      Math.min(box.x1,box.x2), Math.min(box.y1,box.y2),
      Math.max(box.x1,box.x2), Math.max(box.y1,box.y2)
    ] : null,
    points: points,
  };
  fetch('/segment', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload),
  }).then(r=>r.json()).then(data => {
    if (data.error) { status.textContent = 'Error: '+data.error; return; }
    maskB64 = data.mask;
    drawOverlay(data.preview);
    saveBtn.disabled = false;
    keepBtn.disabled = false;
    const npts = points.length;
    status.textContent = `Mask ready. Keep (K) to accumulate, or save directly (S).`;
  });
}

// ── mode ───────────────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  uiCanvas.style.cursor = m==='box' ? 'crosshair' : 'pointer';
  document.getElementById('btnBox').className = m==='box' ? 'mode-active-box' : '';
  document.getElementById('btnFg').className  = m==='fg'  ? 'mode-active-fg'  : '';
  document.getElementById('btnBg').className  = m==='bg'  ? 'mode-active-bg'  : '';
}

// ── undo / clear current ───────────────────────────────────────────────────
function undoLast() {
  if (!history.length) return;
  const last = history.pop();
  if (last.type === 'box') {
    box = last.prev;
  } else if (last.type === 'point') {
    points.pop();
  }
  drawUI();
  if (box || points.length > 0) runSAM();
  else {
    oc.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
    maskB64 = null;
    keepBtn.disabled = true;
    saveBtn.disabled = keptMasks.length === 0;
  }
}

function clearCurrent() {
  box = null; points = []; history = [];
  maskB64 = null;
  keepBtn.disabled = true;
  if (keptMasks.length === 0) saveBtn.disabled = true;
  oc.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
  uc.clearRect(0,0,uiCanvas.width,uiCanvas.height);
  status.textContent = currentFile + ' — draw a box or click to segment';
}

// ── mask accumulation ──────────────────────────────────────────────────────
function keepMask() {
  if (!maskB64) return;
  keptMasks.push(maskB64);
  const n = keptMasks.length;
  document.getElementById('keptCount').textContent = n + ' mask' + (n!==1?'s':'') + ' kept';
  previewBtn.disabled = false;
  saveBtn.disabled = false;
  status.textContent = `Mask ${n} kept! Draw another box or save.`;
  // Clear current prompts ready for next mask
  box = null; points = []; history = []; maskB64 = null;
  keepBtn.disabled = true;
  oc.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
  uc.clearRect(0,0,uiCanvas.width,uiCanvas.height);
}

function clearKept() {
  keptMasks = [];
  document.getElementById('keptCount').textContent = '0 masks kept';
  previewBtn.disabled = true;
  if (!maskB64) saveBtn.disabled = true;
  status.textContent = 'Kept masks cleared.';
}

function previewUnion() {
  if (!keptMasks.length) return;
  // Include current mask in preview if exists
  const allMasks = maskB64 ? [...keptMasks, maskB64] : [...keptMasks];
  status.textContent = 'Computing union...';
  fetch('/union', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({masks: allMasks})
  }).then(r=>r.json()).then(data => {
    drawOverlay(data.preview);
    // Store union as current mask for saving
    maskB64 = data.mask;
    saveBtn.disabled = false;
    status.textContent = `Union of ${allMasks.length} mask${allMasks.length!==1?'s':''}. Save (S) when ready!`;
  });
}

// ── save / nav ─────────────────────────────────────────────────────────────
function saveMask() {
  // If there are kept masks, save their union; otherwise save current mask
  if (keptMasks.length > 0) {
    const allMasks = maskB64 ? [...keptMasks, maskB64] : [...keptMasks];
    fetch('/union', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({masks: allMasks})
    }).then(r=>r.json()).then(data => {
      doSave(data.mask);
    });
  } else if (maskB64) {
    doSave(maskB64);
  }
}

function doSave(mask) {
  fetch('/save', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({file: currentFile, mask: mask})
  }).then(r=>r.json()).then(data => {
    if (data.ok) {
      status.textContent = 'Saved ✓';
      document.getElementById('fi_'+fileIdx).classList.add('done');
      setTimeout(() => nextImage(), 500);
    }
  });
}

function nextImage() {
  if (fileIdx < files.length-1) loadFile(fileIdx+1);
  else status.textContent = 'All done!';
}
function prevImage() {
  if (fileIdx > 0) loadFile(fileIdx - 1);
}

// ── keyboard ───────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key==='b'||e.key==='B') setMode('box');
  if (e.key==='f'||e.key==='F') setMode('fg');
  if (e.key==='x'||e.key==='X') setMode('bg');
  if (e.key==='k'||e.key==='K') keepMask();
  if (e.key==='p'||e.key==='P') previewUnion();
  if (e.key===' ') { e.preventDefault(); runSAM(); }
  if (e.key==='z'||e.key==='Z') undoLast();
  if (e.key==='c'||e.key==='C') clearCurrent();
  if (e.key==='s'||e.key==='S') saveMask();
  if (e.key===',') prevImage();
  if (e.key==='n'||e.key==='N') nextImage();
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML


@app.route("/files")
def list_files():
    files = sorted([
        f.name for f in Path(input_dir).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]) if input_dir else []
    return jsonify({"files": files})


@app.route("/load")
def load_image():
    global current_image
    fname = request.args.get("file")
    path = Path(input_dir) / fname
    img = Image.open(path).convert("RGB")
    current_image = np.array(img)
    predictor.set_image(current_image)
    return jsonify({"image": image_to_base64(img)})


@app.route("/segment", methods=["POST"])
def segment():
    data = request.json
    box = data.get("box")
    pts = data.get("points", [])

    sam_box = np.array(box) if box else None
    sam_points = np.array([[p["x"], p["y"]] for p in pts]) if pts else None
    sam_labels = np.array([p["label"] for p in pts]) if pts else None

    if sam_box is None and sam_points is None:
        return jsonify({"error": "No prompts"})

    masks, scores, _ = predictor.predict(
        point_coords=sam_points,
        point_labels=sam_labels,
        box=sam_box,
        multimask_output=True,
    )

    best = masks[np.argmax(scores)]
    best = robust_fill(best)

    mask_b64 = image_to_base64(
        Image.fromarray((best * 255).astype(np.uint8)), format="PNG"
    )
    preview_b64 = mask_to_overlay(best)

    return jsonify({"mask": mask_b64, "preview": preview_b64})


@app.route("/union", methods=["POST"])
def union():
    data = request.json
    masks_b64 = data["masks"]

    h, w = current_image.shape[:2]
    union_mask = np.zeros((h, w), dtype=bool)

    for mb64 in masks_b64:
        mask_bytes = base64.b64decode(mb64)
        m = np.array(Image.open(BytesIO(mask_bytes)).convert("L")) > 128
        union_mask |= m

    union_mask = robust_fill(union_mask)

    mask_b64 = image_to_base64(
        Image.fromarray((union_mask * 255).astype(np.uint8)), format="PNG"
    )
    preview_b64 = mask_to_overlay(union_mask)

    return jsonify({"mask": mask_b64, "preview": preview_b64})


@app.route("/save", methods=["POST"])
def save():
    data = request.json
    fname = data["file"]
    mask_b64 = data["mask"]
    mask_bytes = base64.b64decode(mask_b64)
    mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L")) > 128
    src = Path(input_dir) / fname
    img = Image.open(src).convert("RGB")
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        img.size, Image.NEAREST
    )
    # Save as RGBA PNG with transparent background
    rgba = img.convert("RGBA")
    rgba.putalpha(mask_pil)
    out = Path(output_dir) / (Path(fname).stem + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out)
    return jsonify({"ok": True, "saved_to": str(out)})


def main():
    global input_dir, output_dir
    parser = argparse.ArgumentParser(description="Interactive SAM masking — box + click + accumulate")
    parser.add_argument("--input-dir", default="data/normalized_flux/central_asian_plov")
    parser.add_argument("--output-dir", default="data/extracted/central_asian_plov")
    parser.add_argument("--sam-weights", default=SAM_WEIGHTS)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    load_sam(device=args.device)

    print(f"\nOpen: http://localhost:{args.port}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()