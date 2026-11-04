"""End-to-end orchestration: imagery -> segmentation -> (agent audit) -> zones -> routes -> brief (+ metrics)."""
import time

import numpy as np

from . import agent as agent_mod
from . import decision, metrics
from .config import get_settings
from .reasoner import build_summary, get_reasoner
from .sam import MetaSAMRefiner
from .segmentation import ClassicalSegmenter


def _finalize_zones(prob, roads, buildings, s, verdicts):
    zones = decision.zone_stats(prob, roads, buildings, s)
    for z in zones:
        v = verdicts.get(z["id"])
        z["verification"] = v["status"] if v else "not_audited"
        if z["verification"] == "unresolved":  # honest uncertainty: widen the range, flag for ground check
            p = z["population"]
            p["lo"], p["hi"] = min(p["lo"], 0.5 * p["exp"]), max(p["hi"], 1.5 * p["exp"])
    return zones, decision.rank(zones)


def analyze(pre, post, *, roads=None, buildings=None, depot=None, truth=None,
            method="change", settings=None, use_meta=True, agent=True):
    s = settings or get_settings()
    t0 = time.perf_counter()
    seg = ClassicalSegmenter()
    prob, perm = seg.flood_prob(pre, post, method)

    sam = MetaSAMRefiner(s)
    sam_info = {"used": False, "error": None, "configured": sam.available()}
    if use_meta and sam.available():
        prob, info = sam.refine(post, prob, perm)
        sam_info.update(info)

    reasoner = get_reasoner(s, use_meta)
    zones, order = _finalize_zones(prob, roads, buildings, s, {})

    trace, verdicts, astats = [], {}, {"zones_audited": 0, "tool_runs": 0, "planner": "off"}
    if agent:
        prob, trace, verdicts, astats = agent_mod.audit(
            pre, post, prob, perm, order, s, sam if use_meta else _NoSAM(),
            planner=getattr(reasoner, "plan_tools", None))
        zones, order = _finalize_zones(prob, roads, buildings, s, verdicts)

    mask = prob >= 0.5
    perm_mask = perm >= 0.5
    top = [z for z in order if z["rank"]][:5]

    if depot is None:  # nearest dry pixel to the bottom-left corner
        dry = np.argwhere(~mask & ~perm_mask)
        depot = tuple(dry[np.argmin((dry[:, 0] - (mask.shape[0] - 1)) ** 2 + dry[:, 1] ** 2)]) if len(dry) else (0, 0)
    routes, depot_used = decision.plan_routes(mask, perm_mask, roads, depot, top, s)
    decision.assign_needs(top, routes)

    px_km2 = (s.pixel_m / 1000.0) ** 2
    totals = {
        "flooded_km2": float(prob.sum() * px_km2),
        "people_exp": sum(z["population"]["exp"] for z in zones),
        "people_lo": sum(z["population"]["lo"] for z in zones),
        "people_hi": sum(z["population"]["hi"] for z in zones),
        "zones_affected": sum(1 for z in zones if z["urgency"] > 0),
    }
    t1 = time.perf_counter()
    brief = reasoner.reason(build_summary(zones, top, routes, totals))
    telemetry = {
        "vision_seconds": round(t1 - t0, 2), "reasoning_seconds": round(time.perf_counter() - t1, 2),
        "meta_chat_calls": getattr(reasoner, "calls", 0), "sam_used": bool(sam_info.get("used")),
        "agent": astats,
    }

    result = {
        "method": method, "totals": totals, "zones": zones, "top": top, "routes": routes,
        "depot": list(depot_used), "brief": brief, "sam": sam_info, "agent_trace": trace,
        "verdicts": verdicts, "telemetry": telemetry,
        "arrays": {"prob": prob, "perm": perm, "mask": mask},
    }
    if truth is not None:
        tz = decision.zone_stats(truth.astype(np.float32), roads, buildings, s)
        t_order = decision.rank(tz)
        result["metrics"] = {
            **metrics.seg_metrics(mask, truth),
            "ece": metrics.ece(prob, truth),
            "top5_overlap": metrics.topk_overlap(order, t_order),
            "interval_coverage": metrics.coverage(zones, tz),
        }
    return result


class _NoSAM:
    def available(self):
        return False
