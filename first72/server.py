"""Zero-dependency web server (stdlib only) serving the UI and a small JSON API."""
import base64
import io
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from . import render
from .config import get_settings
from .imagery import make_scenario
from .pipeline import analyze
from .reasoner import MetaReasoner
from .report import build_html
from .sam import MetaSAMRefiner

WEB = os.path.join(os.path.dirname(__file__), "web")
_cache = {}


def _scenario(seed, difficulty=0):
    key = (seed, difficulty)
    if key not in _cache:
        if len(_cache) > 20:
            _cache.clear()
        _cache[key] = make_scenario(seed, difficulty=difficulty)
    return _cache[key]


def build_payload(res, pre, post, scene, truth=None):
    """JSON-able payload (with embedded PNG layers) shared by the web API, the CLI and the report."""
    s = get_settings()
    a = res["arrays"]
    layers = {
        "overlay": render.data_url(render.overlay(post, a["prob"], a["perm"], res["zones"], res["top"],
                                                  res["routes"], res["depot"], s.grid)),
        "post": render.data_url(post), "pre": render.data_url(pre), "pred": render.data_url(render.mask_img(a["mask"])),
    }
    if truth is not None:
        layers["truth"] = render.data_url(render.mask_img(truth))
    return {
        "method": res["method"], "totals": res["totals"], "top": res["top"], "zones": res["zones"],
        "routes": {k: (v["eta_min"] if v else None) for k, v in res["routes"].items()},
        "brief": res["brief"], "sam": res["sam"], "metrics": res.get("metrics"), "layers": layers,
        "agent_trace": res["agent_trace"], "telemetry": res["telemetry"],
        "grid": s.grid, "pixel_m": s.pixel_m, "scene": scene,
    }


def run_synthetic(seed, method, use_meta=True, agent=True, difficulty=0):
    sc = _scenario(seed, difficulty)
    res = analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                  truth=sc["truth"], method=method, settings=get_settings(), use_meta=use_meta, agent=agent)
    return build_payload(res, sc["pre"], sc["post"], {"synthetic": True, "seed": seed, "difficulty": difficulty}, sc["truth"])


def _decode(data_url, max_side=1024):
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((int(im.width * k), int(im.height * k)))
    return np.array(im)


def run_upload(body):
    pre, post = _decode(body["pre"]), _decode(body["post"])
    if pre.shape != post.shape:
        post = np.array(Image.fromarray(post).resize((pre.shape[1], pre.shape[0])))
    res = analyze(pre, post, method=body.get("method", "change"), settings=get_settings(),
                  use_meta=body.get("meta", True), agent=body.get("agent", True))
    return build_payload(res, pre, post, {"synthetic": False})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/health":
                s = get_settings()
                return self._send(200, {"ok": True, "reasoner": "meta" if MetaReasoner.available(s) else "offline",
                                        "reasoning_model": s.reasoning_model, "sam_configured": MetaSAMRefiner(s).available()})
            if u.path == "/api/scenario":
                return self._send(200, run_synthetic(
                    int(q.get("seed", ["0"])[0]), q.get("method", ["change"])[0], q.get("meta", ["1"])[0] != "0",
                    q.get("agent", ["1"])[0] != "0", int(q.get("difficulty", ["0"])[0])))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/api/analyze":
                return self._send(200, run_upload(body))
            if self.path == "/api/report":  # body = a payload previously returned by this server
                return self._send(200, build_html(body).encode("utf-8"), "text/html; charset=utf-8",
                                  {"Content-Disposition": "attachment; filename=first72_report.html"})
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def serve(host="127.0.0.1", port=8072):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"First72 running at http://{host}:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
