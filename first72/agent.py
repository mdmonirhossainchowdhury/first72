"""Autonomous audit agent: cross-checks the zones the pipeline is least sure about.

Loop (per uncertain top-priority zone):
  1. DIAGNOSE  - ambiguous-pixel share and population-range width flag zones worth a second look.
  2. PLAN      - choose tools. Muse (Meta) can choose the order; the choice is validated against the
                 allowed tool list, and a deterministic default is used if the model fails.
  3. ACT       - run independent evidence tools on the zone crop:
                   alt_index        a second spectral index (blue-red ratio), globally calibrated
                   local_threshold  Otsu recomputed on this zone only (clamped near the global one)
                   sam_reprompt     SAM 3.1 on the zone crop (only if configured)
  4. VERIFY    - the agent does NOT grade itself by sharpening probabilities. It compares independent
                 tool outputs: 'confirmed' (a tool agrees with the baseline mask -> estimate unchanged),
                 'revised' (two tools agree with each other but not the baseline -> adopt their consensus),
                 or 'unresolved' (no agreement -> keep baseline, widen the uncertainty range, flag for
                 ground check). Only 'revised' ever changes the map.
Every step is written to a trace the UI and report display.
"""
import numpy as np

from .segmentation import ClassicalSegmenter, alt_index, water_index

ALLOWED = ("alt_index", "local_threshold", "sam_reprompt")
DEFAULT_PLAN = list(ALLOWED)
AGREE = 0.85


def _iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def diagnose(z, p):
    pop = z["population"]
    return {
        "zone": z["id"], "certainty": round(z["certainty"], 3),
        "ambiguous_frac": round(float(((p > 0.2) & (p < 0.8)).mean()), 3),
        "pop_range_ratio": round((pop["hi"] - pop["lo"]) / max(pop["exp"], 1.0), 2) if pop["exp"] >= 5 else 0.0,
        "flooded_frac": round(z["flooded_frac"], 2),
    }


def needs_audit(d):
    return d["ambiguous_frac"] > 0.03 or d["pop_range_ratio"] > 0.6


def audit(pre, post, prob, perm, order, s, sam, planner=None, max_zones=8):
    seg = ClassicalSegmenter()
    prob = prob.copy()
    trace, verdicts = [], {}
    cands = []
    for z in order:
        if not z["rank"]:
            break
        y0, y1, x0, x1 = z["bbox"]
        d = diagnose(z, prob[y0:y1, x0:x1])
        if needs_audit(d):
            cands.append((z, d))
        if len(cands) >= max_zones:
            break
    stats = {"zones_audited": len(cands), "tool_runs": 0, "planner": "default"}
    if not cands:
        trace.append("No zone exceeded the uncertainty thresholds; nothing to audit.")
        return prob, trace, verdicts, stats

    plans = {}
    if planner:
        got = planner([d for _, d in cands])
        if got:
            plans, stats["planner"] = got, "meta"
            trace.append("Planner: Muse chose the tool order per zone (validated against the allowed tool list).")
    cal = {k: seg.calibrate(img, fn) for k, (img, fn) in {
        "post_alt": (post, alt_index), "pre_alt": (pre, alt_index),
        "post_main": (post, water_index), "pre_main": (pre, water_index)}.items()}

    for z, d in cands:
        y0, y1, x0, x1 = z["bbox"]
        sl = (slice(y0, y1), slice(x0, x1))
        base, est, ran = prob[sl].copy(), {}, []
        for tool in plans.get(z["id"], DEFAULT_PLAN):
            if tool == "alt_index":
                pp = seg.water_prob(post[sl], alt_index, *cal["post_alt"])
                pr = seg.water_prob(pre[sl], alt_index, *cal["pre_alt"])
                est[tool] = pp * (1 - pr)
            elif tool == "local_threshold":
                parts = []
                for img, key in ((post, "post_main"), (pre, "pre_main")):
                    tg, gg = cal[key]
                    tl, _ = seg.calibrate(img[sl], water_index)
                    parts.append(seg.water_prob(img[sl], water_index, float(np.clip(tl, tg - 0.25 * gg, tg + 0.25 * gg)), gg))
                est[tool] = parts[0] * (1 - parts[1])
            elif tool == "sam_reprompt" and sam.available():
                q, info = sam.refine(post[sl], base, perm[sl])
                if not info.get("used"):
                    continue
                est[tool] = q
            else:
                continue
            ran.append(tool)
            stats["tool_runs"] += 1

        m0 = base >= 0.5
        ag = {k: _iou(v >= 0.5, m0) for k, v in est.items()}
        agree = [k for k, v in ag.items() if v >= AGREE]
        if agree:
            status, newp = "confirmed", base  # a confirmation never alters the estimate
        else:
            ks = list(est)
            pairs = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:] if _iou(est[a] >= 0.5, est[b] >= 0.5) >= AGREE]
            if pairs:
                used = sorted({k for pr_ in pairs for k in pr_})
                status, newp = "revised", np.mean([est[k] for k in used], axis=0)
            else:
                status, newp = "unresolved", base
        prob[sl] = newp.astype(np.float32)
        verdicts[z["id"]] = {"status": status, "tools": ran, "agreement": {k: round(v, 3) for k, v in ag.items()}}
        trace.append(
            f"{z['id']}: certainty {d['certainty']:.2f}, {d['ambiguous_frac']:.0%} ambiguous pixels -> ran "
            f"{', '.join(ran) or 'no tools'}; agreement with baseline "
            f"{', '.join(f'{k} {v:.2f}' for k, v in ag.items()) or 'n/a'} -> {status.upper()}"
            + (" (flag for ground check; uncertainty range widened)" if status == "unresolved" else ""))
    return prob, trace, verdicts, stats
