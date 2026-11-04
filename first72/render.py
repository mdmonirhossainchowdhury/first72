"""Image rendering helpers (PNG bytes / data URLs)."""
import base64
import io

import numpy as np
from PIL import Image, ImageDraw


def to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def data_url(arr):
    return "data:image/png;base64," + base64.b64encode(to_png(arr)).decode()


def _scale(n):
    return max(1, 768 // n)


def overlay(post, prob, perm, zones, top, routes, depot, grid):
    H, W = prob.shape
    k = _scale(max(H, W))
    base = post.astype(np.float32)
    tint = np.array([255, 40, 90], np.float32)
    a = (0.55 * prob * (1 - perm))[..., None]
    img = (base * (1 - a) + tint * a).astype(np.uint8)
    im = Image.fromarray(img).resize((W * k, H * k), Image.NEAREST).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    for i in range(1, grid):
        d.line([(0, i * H * k // grid), (W * k, i * H * k // grid)], fill=(255, 255, 255, 70))
        d.line([(i * W * k // grid, 0), (i * W * k // grid, H * k)], fill=(255, 255, 255, 70))
    for zid, r in routes.items():
        if r and len(r["path"]) > 1:
            d.line([(x * k, y * k) for y, x in r["path"]], fill=(255, 220, 0, 230), width=3)
    for z in top:
        y0, y1, x0, x1 = z["bbox"]
        cx, cy = (x0 + x1) // 2 * k, (y0 + y1) // 2 * k
        col = (255, 255, 255, 255) if z["access"] == "road" else (255, 140, 0, 255)
        d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=(20, 20, 30, 220), outline=col, width=2)
        d.text((cx - 4, cy - 6), str(z["rank"]), fill=col)
    dy, dx = depot
    d.rectangle([dx * k - 7, dy * k - 7, dx * k + 7, dy * k + 7], fill=(40, 200, 90, 255), outline=(255, 255, 255, 255))
    return np.array(im.convert("RGB"))


def mask_img(mask):
    return np.stack([mask.astype(np.uint8) * 255] * 3, axis=-1)
