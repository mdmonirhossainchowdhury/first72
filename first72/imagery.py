"""Synthetic pre/post-flood scenes with exact ground truth.

IMPORTANT: metrics computed on these scenes are *synthetic-benchmark* numbers. They prove the
pipeline is wired correctly and let us compare methods; they are NOT real-world accuracy. For
real accuracy, run `python -m first72 analyze` on Sentinel-2 / Sen1Floods11 tiles with labels.
"""
import numpy as np


def _value_noise(rng, n, cells):
    g = rng.random((cells + 2, cells + 2))
    t = np.linspace(0, cells, n, endpoint=False)
    i = t.astype(int)
    f = t - i
    f = f * f * (3 - 2 * f)
    a = g[i, :] * (1 - f)[:, None] + g[i + 1, :] * f[:, None]
    return a[:, i] * (1 - f)[None, :] + a[:, i + 1] * f[None, :]


def _box(a, k=5):
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for dy in range(k):
        for dx in range(k):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / (k * k)


def _draw_road(mask, pts, width=2):
    """pts: list of (y, x) control points; linear interpolation."""
    h, w = mask.shape
    for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
        steps = int(max(abs(y1 - y0), abs(x1 - x0))) * 2 + 1
        for t in np.linspace(0, 1, steps):
            y, x = int(round(y0 + (y1 - y0) * t)), int(round(x0 + (x1 - x0) * t))
            mask[max(0, y - width // 2):min(h, y + width // 2 + 1),
                 max(0, x - width // 2):min(w, x + width // 2 + 1)] = True


def make_scenario(seed=0, size=256, difficulty=0):
    """difficulty=1 adds muddier water, heavier haze/shadow and sensor noise (harder, more realistic)."""
    rng = np.random.default_rng(seed)
    n = size
    h = 0.5 * _value_noise(rng, n, 4) + 0.3 * _value_noise(rng, n, 8) + 0.2 * _value_noise(rng, n, 16)
    yy, xx = np.mgrid[0:n, 0:n] / n
    center = rng.uniform(0.4, 0.6) + rng.uniform(0.08, 0.18) * np.sin(
        2 * np.pi * xx * rng.uniform(1, 2) + rng.uniform(0, 6.28))
    h = 0.5 * h + 1.6 * np.abs(yy - center)

    flood_frac = rng.uniform(0.18, 0.34)
    t1, t2 = np.quantile(h, 0.05), np.quantile(h, 0.05 + flood_frac)
    pre_water, post_water = h < t1, h < t2
    truth = post_water & ~pre_water

    # roads (grid-ish network that crosses the river on bridges)
    roads = np.zeros((n, n), bool)
    ry = int(rng.uniform(0.15, 0.3) * n)
    ry2 = int(rng.uniform(0.7, 0.85) * n)
    rx = int(rng.uniform(0.2, 0.35) * n)
    rx2 = int(rng.uniform(0.65, 0.8) * n)
    _draw_road(roads, [(ry, 0), (ry + 6, n // 2), (ry - 4, n - 1)])
    _draw_road(roads, [(ry2, 0), (ry2 - 8, n // 2), (ry2 + 5, n - 1)])
    _draw_road(roads, [(0, rx), (n // 2, rx + 8), (n - 1, rx - 4)])
    _draw_road(roads, [(0, rx2), (n // 2, rx2 - 6), (n - 1, rx2 + 6)])

    # buildings (footprints with an estimated population each)
    buildings = []
    occupied = roads.copy()
    for _ in range(260):
        bh, bw = int(rng.integers(3, 6)), int(rng.integers(3, 7))
        y, x = int(rng.integers(0, n - bh)), int(rng.integers(0, n - bw))
        sl = (slice(y, y + bh), slice(x, x + bw))
        if occupied[sl].any() or pre_water[sl].any():
            continue
        occupied[sl] = True
        buildings.append({"y": y, "x": x, "h": bh, "w": bw, "pop": int(rng.integers(4, 40))})

    # ---- render ----
    veg = _value_noise(rng, n, 8)[..., None]
    green, tan = np.array([88, 128, 72], float), np.array([150, 125, 85], float)
    land = green * (1 - veg) + tan * veg
    land += rng.normal(0, 5, (n, n, 1))
    turbid = rng.uniform(0, 18) if not difficulty else rng.uniform(25, 55)
    water_c = np.array([58 + turbid, 98 + turbid / 2, 140 - turbid / 2], float)
    road_c, roof_c = np.array([112, 106, 100], float), np.array([168, 160, 156], float)

    def render(flooded, sun):
        img = land.copy() * sun
        img[pre_water] = water_c
        img[roads & ~pre_water] = road_c
        for b in buildings:
            sl = (slice(b["y"], b["y"] + b["h"]), slice(b["x"], b["x"] + b["w"]))
            img[sl] = roof_c * sun
        if flooded is not None:
            fb = flooded
            m = fb[..., None]
            img = np.where(m, water_c, img)
            # flooded roofs stay faintly visible
            for b in buildings:
                sl = (slice(b["y"], b["y"] + b["h"]), slice(b["x"], b["x"] + b["w"]))
                if fb[b["y"] + b["h"] // 2, b["x"] + b["w"] // 2]:
                    img[sl] = 0.65 * water_c + 0.35 * roof_c
        return img

    pre = render(None, 1.0)
    post = render(truth, 1.0)
    # post-only nuisances: cloud shadow + haze (different acquisition conditions)
    shadow = _box((_value_noise(rng, n, 3) > 0.78).astype(np.float32), 9)[..., None]
    haze = _box((_value_noise(rng, n, 3) < 0.15).astype(np.float32), 9)[..., None]
    land_m = (~post_water)[..., None]
    sh_k, hz_k = (0.35, 0.3) if not difficulty else (0.6, 0.55)
    post = np.where(land_m, post * (1 - sh_k * shadow), post)
    post = post * (1 - hz_k * haze) + 200 * hz_k * haze
    sig = 6 if not difficulty else 11
    pre = pre + rng.normal(0, sig, pre.shape)
    post = post + rng.normal(0, sig, post.shape)
    pre, post = np.clip(pre, 0, 255).astype(np.uint8), np.clip(post, 0, 255).astype(np.uint8)

    # depot on high ground, on a road, not flooded
    cand = roads & ~post_water
    hh = np.where(cand, h, -1)
    dy, dx = np.unravel_index(np.argmax(hh), hh.shape)
    return {
        "pre": pre, "post": post, "truth": truth, "roads": roads,
        "buildings": buildings, "depot": (int(dy), int(dx)),
        "meta": {"seed": seed, "size": n, "synthetic": True, "difficulty": difficulty, "flood_fraction": float(truth.mean())},
    }
