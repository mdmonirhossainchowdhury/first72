"""Classical, dependency-light flood segmentation (the always-available baseline + fallback).

Water index for RGB:  (G + B - 2R) / (G + B + 2R).  For real multispectral data swap in NDWI
(green/NIR) or MNDWI (green/SWIR) -- only `water_index` changes. `alt_index` is an independent
second index (blue-red ratio) used by the audit agent to cross-check uncertain zones.
"""
import numpy as np


def water_index(img):
    r, g, b = (img[..., i].astype(np.float32) for i in range(3))
    return (g + b - 2 * r) / (g + b + 2 * r + 1e-6)


def alt_index(img):
    r, b = img[..., 0].astype(np.float32), img[..., 2].astype(np.float32)
    return (b - r) / (b + r + 1e-6)


def box3(a):
    p = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / 9.0


def otsu(x, bins=256):
    hist, edges = np.histogram(x, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def calib(idx):
    """(threshold, class-mean gap) of an already-smoothed index image."""
    thr = otsu(idx)
    hi, lo = idx[idx >= thr], idx[idx < thr]
    gap = (hi.mean() - lo.mean()) if hi.size and lo.size else 0.2
    return thr, max(float(gap), 1e-3)


class ClassicalSegmenter:
    name = "classical-water-index"

    def calibrate(self, img, idx_fn=water_index):
        return calib(box3(idx_fn(img)))

    def water_prob(self, img, idx_fn=water_index, thr=None, gap=None):
        idx = box3(idx_fn(img))
        t, g = calib(idx)
        t = t if thr is None else thr
        g = g if gap is None else gap
        return _sigmoid((idx - t) / max(0.01, g / 8.0)).astype(np.float32)

    def flood_prob(self, pre, post, method="change"):
        """Returns (flood_prob, permanent_water_prob).

        method="change":    flooded = water in POST and not water in PRE   (the real method)
        method="post_only": flooded = water in POST                        (naive baseline)
        """
        pp = self.water_prob(post)
        pr = self.water_prob(pre)
        if method == "post_only":
            return pp, pr
        return (pp * (1 - pr)).astype(np.float32), pr
