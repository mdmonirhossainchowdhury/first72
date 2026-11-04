"""Decision layer: zone statistics with uncertainty ranges, urgency ranking, and flood-aware routing."""
import heapq
import math

import numpy as np


def zone_id(r, c):
    return f"{chr(65 + r)}{c + 1}"


def zone_stats(prob, roads, buildings, s):
    """Per-zone expected / low / high estimates derived from pixel probabilities.

    low  = only pixels we are >80% sure are flooded;  high = everything >20% likely flooded.
    """
    H, W = prob.shape
    g = s.grid
    ys, xs = np.linspace(0, H, g + 1).astype(int), np.linspace(0, W, g + 1).astype(int)
    px_km2 = (s.pixel_m / 1000.0) ** 2
    roads = roads if roads is not None else np.zeros_like(prob, bool)

    bz = {}
    if buildings:
        for b in buildings:
            cy, cx = b["y"] + b["h"] // 2, b["x"] + b["w"] // 2
            r = min(g - 1, int(np.searchsorted(ys, cy, side="right") - 1))
            c = min(g - 1, int(np.searchsorted(xs, cx, side="right") - 1))
            bz.setdefault((r, c), []).append((cy, cx, b["pop"]))

    zones = []
    for r in range(g):
        for c in range(g):
            sl = (slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1]))
            p = prob[sl]
            flooded = p >= 0.5
            area = {"exp": float(p.sum() * px_km2), "lo": float((p > 0.8).sum() * px_km2),
                    "hi": float((p > 0.2).sum() * px_km2)}
            if buildings is not None:
                items = bz.get((r, c), [])
                if items:
                    cy = np.array([i[0] for i in items]); cx = np.array([i[1] for i in items])
                    pop = np.array([i[2] for i in items], float)
                    pp = prob[cy, cx]
                    population = {"exp": float((pop * pp).sum()), "lo": float((pop * (pp > 0.8)).sum()),
                                  "hi": float((pop * (pp > 0.2)).sum())}
                    b_total, b_flood = len(items), int((pp >= 0.5).sum())
                else:
                    population, b_total, b_flood = {"exp": 0.0, "lo": 0.0, "hi": 0.0}, 0, 0
            else:  # no footprints: assume uniform density
                d = s.default_density
                population = {k: area[k] * d for k in ("exp", "lo", "hi")}
                b_total, b_flood = 0, 0
            for d in (area, population):  # a range must always bracket its own estimate
                d["lo"], d["hi"] = min(d["lo"], d["exp"]), max(d["hi"], d["exp"])
            rz = roads[sl]
            rb = int((rz & flooded).sum())
            zones.append({
                "id": zone_id(r, c), "row": r, "col": c,
                "bbox": [int(ys[r]), int(ys[r + 1]), int(xs[c]), int(xs[c + 1])],
                "flooded_frac": float(flooded.mean()),
                "area_km2": area, "population": population,
                "buildings_total": b_total, "buildings_flooded": b_flood,
                "road_px": int(rz.sum()), "road_blocked_frac": float(rb / rz.sum()) if rz.sum() else 0.0,
                "certainty": float(np.maximum(p, 1 - p).mean()),
            })
    return zones


def rank(zones):
    mp = max((z["population"]["exp"] for z in zones), default=0) + 1e-9
    mb = max((z["buildings_flooded"] for z in zones), default=0) + 1e-9
    for z in zones:
        u = 100 * (0.55 * z["population"]["exp"] / mp + 0.20 * z["flooded_frac"]
                   + 0.15 * z["buildings_flooded"] / mb + 0.10 * z["road_blocked_frac"])
        z["urgency"] = float(u) if z["flooded_frac"] >= 0.02 else 0.0
    order = sorted(zones, key=lambda z: -z["urgency"])
    for i, z in enumerate(order, 1):
        z["rank"] = i if z["urgency"] > 0 else None
    return order


def plan_routes(flood_mask, perm_mask, roads, depot, zones, s, topk=5):
    """Dijkstra over a coarse cost grid. Flooded cells are impassable; permanent-water cells are
    passable only where a road crosses (bridge). Zones with no reachable dry cell are marked
    'boat_or_air'. Travel time is a ROUGH estimate (roads at road_speed_kmh, off-road 3x slower)."""
    H, W = flood_mask.shape
    cs = max(1, min(H, W) // 64)
    h, w = H // cs, W // cs

    def cellmean(a):
        return a[:h * cs, :w * cs].reshape(h, cs, w, cs).mean(axis=(1, 3))

    ff, pf = cellmean(flood_mask.astype(np.float32)), cellmean(perm_mask.astype(np.float32))
    rf = cellmean((roads if roads is not None else np.zeros_like(flood_mask)).astype(np.float32))
    road_c = rf > 0.1
    passable = (ff < 0.5) & ((pf < 0.5) | road_c)
    cost = np.where(road_c, 1.0, 3.0)

    d0 = (min(h - 1, depot[0] // cs), min(w - 1, depot[1] // cs))
    if not passable[d0]:
        pts = np.argwhere(passable)
        if not len(pts):
            return {z["id"]: None for z in zones[:topk]}, depot
        d0 = tuple(pts[np.argmin(((pts - np.array(d0)) ** 2).sum(1))])

    dist = np.full((h, w), np.inf)
    prev = {}
    dist[d0] = 0.0
    pq = [(0.0, d0)]
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while pq:
        dcur, (y, x) = heapq.heappop(pq)
        if dcur > dist[y, x]:
            continue
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and passable[ny, nx]:
                nd = dcur + cost[ny, nx] * (1.414 if dy and dx else 1.0)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(ny, nx)] = (y, x)
                    heapq.heappush(pq, (nd, (ny, nx)))

    m_per_min = s.road_speed_kmh * 1000 / 60
    out = {}
    for z in zones[:topk]:
        y0, y1, x0, x1 = z["bbox"]
        sub = dist[y0 // cs:max(y0 // cs + 1, y1 // cs), x0 // cs:max(x0 // cs + 1, x1 // cs)]
        if not np.isfinite(sub).any():
            out[z["id"]] = None
            continue
        iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
        cell = (y0 // cs + iy, x0 // cs + ix)
        path, cur = [cell], cell
        while cur in prev:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        minutes = float(dist[cell]) * cs * s.pixel_m / m_per_min
        out[z["id"]] = {"eta_min": int(5 * math.ceil(minutes / 5)) if minutes > 0 else 0,
                        "path": [[int(py * cs + cs // 2), int(px * cs + cs // 2)] for py, px in path]}
    return out, (int(d0[0] * cs + cs // 2), int(d0[1] * cs + cs // 2))


def assign_needs(zones_top, routes):
    for z in zones_top:
        pop, frac = z["population"]["exp"], z["flooded_frac"]
        r = routes.get(z["id"])
        z["access"] = "road" if r else "boat_or_air"
        z["eta_min"] = r["eta_min"] if r else None
        needs = []
        if pop >= 20: needs.append("water")
        if frac >= 0.4 and pop >= 20: needs.append("shelter")
        if pop >= 100: needs.append("food")
        if pop >= 150: needs.append("medical_triage")
        if not r: needs.append("boats")
        z["needs"] = needs
