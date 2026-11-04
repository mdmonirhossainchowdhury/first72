"""`python -m first72 check-meta`: prove (or diagnose) the live Meta integrations in one command.

Run it once after you fill in .env. It makes one tiny reasoning call and one SAM call on a small synthetic image,
prints PASS/FAIL with a plain-English hint, and writes meta_check.json (no secrets) you can keep as evidence that
the repo has a real, functional Meta integration (judging stage one).
"""
import json
import time
import urllib.error

from .config import get_settings
from .imagery import make_scenario
from .reasoner import MetaReasoner
from .sam import MetaSAMRefiner
from .segmentation import ClassicalSegmenter


def hint(exc):
    msg = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, urllib.error.HTTPError):
        c = exc.code
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001
            pass
        tips = {401: "API key rejected: check the key and that it belongs to this service.",
                403: "Not allowed: key lacks access to this model or endpoint.",
                404: "URL or model id not found: check the base URL (and /chat/completions or /responses path) and model name.",
                429: "Rate limited or out of credits: check billing / credits.",
                }
        return f"HTTP {c}. {tips.get(c, 'See the response body below.')} {body}"
    if "meta_sam_parser" in msg:
        return msg + "  -> run: pip install meta-sam-parser"
    if isinstance(exc, urllib.error.URLError):
        return msg + "  -> cannot reach the host: check the base URL and your internet connection."
    return msg


def run(settings=None, out="meta_check.json"):
    s = settings or get_settings()
    results = []

    def rec(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        results.append({"check": name, "pass": bool(ok), "detail": detail})

    print(f"reasoning model: {s.reasoning_model} | reasoning base: {s.api_base or '(not set)'}")
    print(f"SAM model: {s.sam_model} | SAM base: {s.sam_endpoint or s.sam_api_base or '(not set)'} | mode: {s.sam_mode}\n")

    # 1) reasoning model (Muse Spark / Muse Spark Open)
    if not MetaReasoner.available(s):
        rec("reasoning model", False, "META_API_BASE and META_API_KEY are not both set in .env")
    else:
        try:
            t = time.perf_counter()
            ans = MetaReasoner(s)._chat({"ping": 1}, 'Reply with exactly this JSON object and nothing else: {"ok": true}')
            rec("reasoning model", isinstance(ans, dict), f"answered in {time.perf_counter() - t:.1f}s with {json.dumps(ans)[:80]}")
        except Exception as e:  # noqa: BLE001
            rec("reasoning model", False, hint(e))

    # 2) SAM 3.1 on a small synthetic flood scene
    sam = MetaSAMRefiner(s)
    if not s.use_sam:
        rec("SAM 3.1", True, "SKIPPED on purpose (META_USE_SAM=0, open-weights-only mode: classical segmentation is used). Do not claim SAM in your submission.")
    elif not sam.available():
        rec("SAM 3.1", False, "META_SAM_API_BASE/META_SAM_API_KEY (or META_API_BASE/META_API_KEY) are not set in .env")
    else:
        sc = make_scenario(0, size=256)
        prob, perm = ClassicalSegmenter().flood_prob(sc["pre"], sc["post"])
        t = time.perf_counter()
        _, info = sam.refine(sc["post"], prob, perm)
        if info.get("used"):
            rec("SAM 3.1", True, f"{info.get('masks')} masks decoded in {time.perf_counter() - t:.1f}s (mode={info['mode']})")
        else:
            rec("SAM 3.1", False, hint(Exception(info.get("error") or "no masks returned")) if info.get("error") else "no masks returned")

    ok = all(r["pass"] for r in results)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"checked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "reasoning_model": s.reasoning_model,
                   "sam_model": s.sam_model, "sam_mode": s.sam_mode, "all_passed": ok, "results": results}, f, indent=2)
    print(f"\n{'ALL LIVE META CHECKS PASSED' if ok else 'SOME CHECKS FAILED: fix the items above, then re-run'}  (wrote {out})")
    return ok
