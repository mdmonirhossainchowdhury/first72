"""Evaluate on YOUR real tiles (e.g. exported Sen1Floods11 / Sentinel-2 chips).

Folder layout (any image format Pillow reads; 8-bit or 16-bit):
    <name>_post.png    flood-time image                       (required)
    <name>_label.png   ground-truth water/flood mask, white = flooded   (required)
    <name>_pre.png     pre-flood image of the same tile       (optional; without it, post-only is used)

Prints per-tile and mean IoU / precision / recall / calibration error and writes real_results.md.
Report these numbers as-is, including the bad tiles.
"""
import glob
import os

import numpy as np
from PIL import Image

from .config import get_settings
from .pipeline import analyze


def load_rgb(path, max_side=1024):
    im = Image.open(path)
    a = np.array(im)
    if a.dtype != np.uint8:  # 16-bit etc: percentile-stretch to 8-bit
        a = a.astype(np.float32)
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(a)
    im = im.convert("RGB")
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((int(im.width * k), int(im.height * k)))
    return np.array(im)


def find_tiles(folder):
    tiles = []
    for post in sorted(glob.glob(os.path.join(folder, "*_post.*"))):
        name = os.path.basename(post).rsplit("_post.", 1)[0]
        lab = glob.glob(os.path.join(folder, f"{name}_label.*"))
        pre = glob.glob(os.path.join(folder, f"{name}_pre.*"))
        if lab:
            tiles.append((name, post, pre[0] if pre else None, lab[0]))
    return tiles


def run(folder, out="real_results.md", agent=True):
    tiles = find_tiles(folder)
    if not tiles:
        raise SystemExit(f"No tiles found in {folder}. Expected <name>_post.*, <name>_label.* (and optionally <name>_pre.*).")
    s = get_settings()
    rows = []
    for name, post_p, pre_p, lab_p in tiles:
        post = load_rgb(post_p)
        pre = load_rgb(pre_p) if pre_p else post
        if pre.shape != post.shape:
            pre = np.array(Image.fromarray(pre).resize((post.shape[1], post.shape[0])))
        lab = np.array(Image.open(lab_p).convert("L").resize((post.shape[1], post.shape[0]), Image.NEAREST)) > 127
        method = "change" if pre_p else "post_only"
        r = analyze(pre, post, truth=lab, method=method, settings=s, agent=agent)
        m = r["metrics"]
        rows.append((name, method, m, r["telemetry"]["agent"]))
        print(f"{name:24s} {method:9s} IoU {m['iou']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  ECE {m['ece']:.3f}")
    mean = {k: float(np.mean([r[2][k] for r in rows])) for k in ("iou", "precision", "recall", "ece")}
    lines = [f"# Real-tile evaluation ({len(rows)} tiles)", "",
             "| Tile | Method | IoU | Precision | Recall | ECE |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {me} | {m['iou']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['ece']:.3f} |" for n, me, m, _ in rows]
    lines += [f"| **mean** | | **{mean['iou']:.3f}** | **{mean['precision']:.3f}** | **{mean['recall']:.3f}** | **{mean['ece']:.3f}** |"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nmean IoU {mean['iou']:.3f}  -> wrote {out}")
    return rows
