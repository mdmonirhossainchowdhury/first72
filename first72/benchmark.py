"""Synthetic benchmark: naive baseline vs change detection vs change detection + audit agent."""
import json
import os
from collections import Counter

import numpy as np

from .config import get_settings
from .imagery import make_scenario
from .pipeline import analyze

KEYS = ["iou", "precision", "recall", "f1", "ece", "top5_overlap", "interval_coverage"]
METHODS = [("post_only", "post_only", False), ("change", "change", False), ("change+agent", "change", True)]
NAMES = {"post_only": "Post-only water threshold (baseline)", "change": "Pre/post change detection",
         "change+agent": "Change detection + audit agent (First72)"}


def run(n=30, out_dir="benchmark"):
    s = get_settings()
    out, lines = {}, [f"# First72 synthetic benchmark ({n} scenes per difficulty)", "",
                      "> Synthetic scenes with exact ground truth. These numbers validate the pipeline and compare methods; "
                      "they are **not** real-world accuracy. Re-run on real tiles with `python -m first72 evaluate`.", ""]
    for diff, label in ((0, "Easy scenes"), (1, "Hard scenes (muddy water, heavy haze/shadow, more noise)")):
        rows = {m[0]: [] for m in METHODS}
        verdicts = Counter()
        zone_iou = {"confirmed": [], "revised": [], "unresolved": []}
        for seed in range(n):
            sc = make_scenario(seed, difficulty=diff)
            for name, method, ag in METHODS:
                r = analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                            truth=sc["truth"], method=method, settings=s, use_meta=False, agent=ag)
                rows[name].append(r["metrics"])
                if ag:
                    verdicts.update(v["status"] for v in r["verdicts"].values())
                    m = r["arrays"]["mask"]
                    for z in r["zones"]:
                        if z["verification"] in zone_iou:
                            y0, y1, x0, x1 = z["bbox"]
                            pm, tm = m[y0:y1, x0:x1], sc["truth"][y0:y1, x0:x1]
                            u = (pm | tm).sum()
                            zone_iou[z["verification"]].append((pm & tm).sum() / u if u else 1.0)
        summ = {m: {k: (float(np.nanmean([r[k] for r in rs])), float(np.nanstd([r[k] for r in rs]))) for k in KEYS}
                for m, rs in rows.items()}
        zsum = {k: (len(v), float(np.mean(v)) if v else None) for k, v in zone_iou.items()}
        out[label] = {"summary": summ, "per_scene": rows, "agent_verdicts": dict(verdicts), "zone_iou_by_verdict": zsum}
        lines += [f"## {label}", "", "| Method | IoU | Precision | Recall | F1 | ECE (lower=better) | Top-5 zone overlap | Interval coverage |",
                  "|---|---|---|---|---|---|---|---|"]
        for m, _, _ in METHODS:
            lines.append(f"| {NAMES[m]} | " + " | ".join(f"{summ[m][k][0]:.3f} ± {summ[m][k][1]:.3f}" for k in KEYS) + " |")
        lines += ["", "Does the agent's verdict tell you where the map is wrong? Mean zone-level IoU vs ground truth of the audited zones, by verdict:", "",
                  "| Verdict | Zones | Mean zone IoU |", "|---|---|---|"]
        for k, (cnt, mi) in zsum.items():
            lines.append(f"| {k} | {cnt} | {'n/a' if mi is None else f'{mi:.3f}'} |")
        lines.append("")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"n_scenes": n, "results": out}, f, indent=2)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out, "\n".join(lines)
