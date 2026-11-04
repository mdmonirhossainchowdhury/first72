import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def main(argv=None):
    ap = argparse.ArgumentParser(prog="first72", description="First72 disaster damage-to-aid copilot")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8072)

    p = sub.add_parser("demo", help="run one synthetic scenario and write outputs")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="out")
    p.add_argument("--difficulty", type=int, default=0, choices=[0, 1])
    p.add_argument("--method", default="change", choices=["change", "post_only"])
    p.add_argument("--no-agent", action="store_true")

    p = sub.add_parser("benchmark", help="synthetic benchmark: baseline vs change detection vs +agent")
    p.add_argument("--n", type=int, default=30); p.add_argument("--out", default="benchmark")

    p = sub.add_parser("analyze", help="analyze your own pre/post images")
    p.add_argument("--pre", required=True); p.add_argument("--post", required=True)
    p.add_argument("--truth", help="optional ground-truth flood mask (white = flooded) to compute metrics")
    p.add_argument("--out", default="out_real"); p.add_argument("--no-agent", action="store_true")
    p.add_argument("--method", default="change", choices=["change", "post_only"])

    p = sub.add_parser("evaluate", help="evaluate on a folder of real tiles (<name>_post/_label[/_pre])")
    p.add_argument("--dir", required=True); p.add_argument("--out", default="real_results.md")
    p.add_argument("--no-agent", action="store_true")

    p = sub.add_parser("check-meta", help="test your live Meta API setup (reasoning model + SAM) and write meta_check.json")
    p.add_argument("--out", default="meta_check.json")

    sub.add_parser("reset-budget", help="reset the live API call counter (.api_calls)")

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from .server import serve
        serve(a.host, a.port)
    elif a.cmd == "demo":
        from .config import get_settings
        from .imagery import make_scenario
        from .pipeline import analyze
        s = get_settings(); sc = make_scenario(a.seed, difficulty=a.difficulty)
        r = analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                    truth=sc["truth"], method=a.method, settings=s, agent=not a.no_agent)
        _write(a.out, sc["pre"], sc["post"], r, {"synthetic": True, "seed": a.seed}, sc["truth"])
        print(r["brief"]["en"]); print("\nagent trace:"); [print(" ", x) for x in r["agent_trace"]]
        print("\nmetrics (SYNTHETIC scene):", json.dumps(r["metrics"], indent=2))
    elif a.cmd == "benchmark":
        from .benchmark import run
        print(run(a.n, a.out)[1])
    elif a.cmd == "analyze":
        from .config import get_settings
        from .pipeline import analyze
        from .realeval import load_rgb
        pre, post = load_rgb(a.pre), load_rgb(a.post)
        if pre.shape != post.shape:
            post = np.array(Image.fromarray(post).resize((pre.shape[1], pre.shape[0])))
        truth = None
        if a.truth:
            truth = np.array(Image.open(a.truth).convert("L").resize((pre.shape[1], pre.shape[0]), Image.NEAREST)) > 127
        s = get_settings()
        r = analyze(pre, post, truth=truth, method=a.method, settings=s, agent=not a.no_agent)
        _write(a.out, pre, post, r, {"synthetic": False}, truth)
        print(r["brief"]["en"]); [print(" ", x) for x in r["agent_trace"]]
        if truth is not None:
            print("\nmetrics:", json.dumps(r["metrics"], indent=2))
    elif a.cmd == "reset-budget":
        from . import budget
        budget.reset(); print("API call counter reset")
    elif a.cmd == "check-meta":
        from .check import run as check
        return 0 if check(out=a.out) else 2
    elif a.cmd == "evaluate":
        from .realeval import run
        run(a.dir, a.out, agent=not a.no_agent)
    else:
        ap.print_help()
        return 1
    return 0


def _write(out, pre, post, r, scene, truth):
    from .report import build_html
    from .server import build_payload
    os.makedirs(out, exist_ok=True)
    payload = build_payload(r, pre, post, scene, truth)
    import base64
    with open(os.path.join(out, "overlay.png"), "wb") as f:
        f.write(base64.b64decode(payload["layers"]["overlay"].split(",", 1)[1]))
    with open(os.path.join(out, "report.html"), "w", encoding="utf-8") as f:
        f.write(build_html(payload))
    with open(os.path.join(out, "brief_en.txt"), "w", encoding="utf-8") as f: f.write(r["brief"]["en"])
    with open(os.path.join(out, "brief_bn.txt"), "w", encoding="utf-8") as f: f.write(r["brief"]["bn"])
    slim = {k: v for k, v in r.items() if k != "arrays"}
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False, default=float)
    print(f"wrote {out}/overlay.png, report.html, brief_en.txt, brief_bn.txt, result.json")


if __name__ == "__main__":
    sys.exit(main())
