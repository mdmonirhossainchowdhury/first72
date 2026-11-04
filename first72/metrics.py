"""Evaluation: segmentation quality, calibration, and decision-level agreement."""
import numpy as np


def seg_metrics(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    iou = tp / max(1, tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1}


def ece(prob, truth, bins=10):
    """Expected calibration error of the flood probability (lower = confidence is trustworthy)."""
    p, t = prob.ravel(), truth.ravel().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total, err = len(p), 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.any():
            err += m.sum() / total * abs(p[m].mean() - t[m].mean())
    return float(err)


def coverage(zones, truth_zones):
    """How often the true population falls inside our [lo, hi] range (honesty of the intervals)."""
    hit = n = 0
    for z, t in zip(zones, truth_zones):
        tp = t["population"]["exp"]
        if tp <= 0 and z["population"]["hi"] <= 0:
            continue
        n += 1
        hit += z["population"]["lo"] - 1e-6 <= tp <= z["population"]["hi"] + 1e-6
    return hit / n if n else float("nan")


def topk_overlap(order_pred, order_true, k=5):
    a = {z["id"] for z in order_pred[:k] if z["urgency"] > 0}
    b = {z["id"] for z in order_true[:k] if z["urgency"] > 0}
    return len(a & b) / max(1, len(b))
