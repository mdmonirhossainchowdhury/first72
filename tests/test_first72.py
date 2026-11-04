import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

from first72 import decision
from first72.config import Settings
from first72.imagery import make_scenario
from first72.metrics import ece, seg_metrics
from first72.pipeline import analyze
from first72.reasoner import MetaReasoner, OfflineReasoner, bn


def run(seed=0, method="change", settings=None, use_meta=False):
    sc = make_scenario(seed)
    return sc, analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                       truth=sc["truth"], method=method, settings=settings or Settings(), use_meta=use_meta)


class Core(unittest.TestCase):
    def test_scenario_is_deterministic(self):
        a, b = make_scenario(3), make_scenario(3)
        self.assertTrue((a["post"] == b["post"]).all())
        self.assertTrue((a["truth"] == b["truth"]).all())

    def test_change_detection_beats_baseline_and_is_accurate(self):
        _, ch = run(1, "change")
        _, po = run(1, "post_only")
        self.assertGreater(ch["metrics"]["iou"], 0.9)
        self.assertGreater(ch["metrics"]["iou"], po["metrics"]["iou"])

    def test_zones_and_ranking(self):
        _, r = run(2)
        self.assertEqual(len(r["zones"]), 64)
        self.assertTrue(r["top"])
        self.assertEqual([z["rank"] for z in r["top"]], sorted(z["rank"] for z in r["top"]))
        for z in r["top"]:
            p = z["population"]
            self.assertLessEqual(p["lo"], p["exp"] + 1e-6)
            self.assertLessEqual(p["exp"], p["hi"] + 1e-6)

    def test_unreachable_zones_flagged_and_reachable_have_paths(self):
        _, r = run(0)
        for z in r["top"]:
            if z["access"] == "road":
                self.assertTrue(r["routes"][z["id"]]["path"])
            else:
                self.assertIn("boats", z["needs"])

    def test_metrics_sane(self):
        m = seg_metrics(np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 1], bool))
        self.assertAlmostEqual(m["iou"], 1 / 3)
        p = np.array([0.0, 1.0, 1.0, 0.0]); t = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(ece(p, t), 0.0)

    def test_offline_brief_has_bangla_digits(self):
        _, r = run(0)
        self.assertIn("ফার্স্ট৭২", r["brief"]["bn"])
        self.assertEqual(bn(2025), "২০২৫")


try:
    import meta_sam_parser  # optional dependency (pip install meta-sam-parser)
    HAVE_PARSER = True
except ImportError:
    HAVE_PARSER = False
needs_parser = unittest.skipUnless(HAVE_PARSER, "meta-sam-parser not installed")

SAMPLE = ("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
          "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
          ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
          "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n")  # from meta-sam-parser docs


class FakeMeta(BaseHTTPRequestHandler):
    mode = "good"
    last_sam_text = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n))
        if self.path.endswith("/chat/completions") and body["messages"][0]["content"].startswith("You are a tool planner"):
            ids = [z["zone"] for z in json.loads(body["messages"][1]["content"])["zones"]]
            plan = {z: (["bogus_tool"] if FakeMeta.mode == "bad" and i == 0 else ["local_threshold", "alt_index"]) for i, z in enumerate(ids)}
            out = {"choices": [{"message": {"content": json.dumps({"plans": plan})}}]}
        elif self.path.endswith("/chat/completions") and "priority_zones" not in body["messages"][1]["content"]:
            out = {"choices": [{"message": {"content": '{"ok": true}'}}]}  # check-meta ping
        elif self.path.endswith("/chat/completions"):
            zones = json.loads(body["messages"][1]["content"])["priority_zones"]
            z = zones[0]["zone"] if FakeMeta.mode == "good" else "Z9"
            content = json.dumps({"en_brief": f"Zone {z} is the top priority with the highest exposure; verify on the ground.",
                                  "bn_brief": f"এলাকা {z} সর্বোচ্চ অগ্রাধিকার; মাঠপর্যায়ে যাচাই করুন।",
                                  "actions": [{"zone": z, "action": "water", "why": "most people"}]})
            out = {"choices": [{"message": {"content": "```json\n" + content + "\n```"}}]}
        else:  # SAM via Responses API
            parts = body["input"][0]["content"]
            FakeMeta.last_sam_text = [p for p in parts if p["type"] == "input_text"][0]["text"]
            out = {"output": [{"content": [{"type": "output_text", "text": SAMPLE}]}]}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


class MetaPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeMeta)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def settings(self, **kw):
        return Settings(api_base=self.base, api_key="k", sam_api_base=self.base, sam_api_key="k", **kw)

    def test_meta_reasoner_accepted_when_verified(self):
        FakeMeta.mode = "good"
        _, r = run(0, settings=self.settings(), use_meta=True)
        self.assertTrue(r["brief"]["source"].startswith("meta:"))
        self.assertEqual(r["brief"]["notes"], [])

    def test_meta_reasoner_rejected_when_hallucinating_zone(self):
        FakeMeta.mode = "bad"
        _, r = run(0, settings=self.settings(), use_meta=True)
        self.assertEqual(r["brief"]["source"], OfflineReasoner.source)
        self.assertTrue(r["brief"]["notes"])

    @needs_parser
    def test_decoder_on_official_sample(self):
        from first72.sam import parse_sam_text
        masks = parse_sam_text(SAMPLE, (320, 334))
        self.assertEqual(len(masks), 2)
        for m, conf in masks:
            self.assertEqual(m.shape, (334, 320)); self.assertGreater(m.sum(), 0)

    @needs_parser
    def test_sam_concept_mode_end_to_end_through_http(self):
        _, r = run(0, settings=self.settings(sam_mode="concept"), use_meta=True)
        self.assertTrue(r["sam"]["used"], r["sam"]); self.assertEqual(r["sam"]["masks"], 2)
        self.assertEqual(FakeMeta.last_sam_text, "flooded land")
        self.assertGreater(r["metrics"]["iou"], 0.9)

    @needs_parser
    def test_sam_points_mode_sends_official_object_prompt(self):
        _, r = run(0, settings=self.settings(sam_mode="points"), use_meta=True)
        self.assertTrue(r["sam"]["used"], r["sam"])
        self.assertIn("<|point;", FakeMeta.last_sam_text); self.assertIn("-<|point;", FakeMeta.last_sam_text)

    def test_sam_failure_falls_back_to_classical(self):
        bad = Settings(sam_api_base="http://127.0.0.1:9", sam_api_key="k", timeout_s=2)
        _, r = run(0, settings=bad, use_meta=True)
        self.assertFalse(r["sam"]["used"]); self.assertTrue(r["sam"]["error"])
        self.assertGreater(r["metrics"]["iou"], 0.9)

    def test_check_meta_command(self):
        import os, tempfile
        from first72.check import run as check
        with tempfile.TemporaryDirectory() as d:
            ok = check(self.settings(sam_mode="concept"), out=f"{d}/m.json")
            data = json.load(open(f"{d}/m.json"))
            self.assertEqual(ok, HAVE_PARSER)  # SAM check needs the optional parser; reasoning check always passes
            self.assertTrue(next(r for r in data["results"] if r["check"] == "reasoning model")["pass"])
            self.assertNotIn("\"k\"", json.dumps(data))  # no key material in the evidence file
        with tempfile.TemporaryDirectory() as d:  # nothing configured -> clear FAIL, non-crashing
            self.assertFalse(check(Settings(), out=f"{d}/m.json"))

    def test_verifier_rules(self):
        summ = {"priority_zones": [{"zone": "A1"}, {"zone": "B2"}, {"zone": "C3"}, {"zone": "D4"}]}
        ok = {"en_brief": "A1 first then B2 .......", "bn_brief": "এলাকা A1 প্রথম ..........", "actions": [{"zone": "A1"}]}
        self.assertEqual(MetaReasoner.verify(ok, summ), [])
        self.assertTrue(MetaReasoner.verify({**ok, "actions": [{"zone": "D4"}]}, summ))


class AgentAndTools(unittest.TestCase):
    def test_ranges_always_bracket_estimates(self):
        for diff in (0, 1):
            sc = make_scenario(1, difficulty=diff)
            r = analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                        truth=sc["truth"], settings=Settings(), use_meta=False)
            tt = r["totals"]
            self.assertLessEqual(tt["people_lo"], tt["people_exp"] + 1e-6); self.assertLessEqual(tt["people_exp"], tt["people_hi"] + 1e-6)
            for z in r["zones"]:
                p = z["population"]; self.assertLessEqual(p["lo"], p["exp"] + 1e-6); self.assertLessEqual(p["exp"], p["hi"] + 1e-6)

    def test_confirmation_never_changes_the_map(self):
        sc = make_scenario(0)
        kw = dict(roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"], truth=sc["truth"], settings=Settings(), use_meta=False)
        a, b = analyze(sc["pre"], sc["post"], agent=False, **kw), analyze(sc["pre"], sc["post"], agent=True, **kw)
        if all(v["status"] == "confirmed" for v in b["verdicts"].values()):
            self.assertTrue((a["arrays"]["mask"] == b["arrays"]["mask"]).all())
        self.assertEqual(b["metrics"]["iou"] >= a["metrics"]["iou"] - 0.01, True)

    def test_agent_flags_hard_scene_and_flags_are_informative(self):
        zi = {"confirmed": [], "unresolved": []}
        for seed in range(6):
            sc = make_scenario(seed, difficulty=1)
            r = analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                        truth=sc["truth"], settings=Settings(), use_meta=False)
            m = r["arrays"]["mask"]
            for z in r["zones"]:
                if z["verification"] in zi:
                    y0, y1, x0, x1 = z["bbox"]; pm, tm = m[y0:y1, x0:x1], sc["truth"][y0:y1, x0:x1]; u = (pm | tm).sum()
                    zi[z["verification"]].append((pm & tm).sum() / u if u else 1.0)
        self.assertTrue(zi["unresolved"] and zi["confirmed"])
        self.assertLess(np.mean(zi["unresolved"]), np.mean(zi["confirmed"]) - 0.15)  # flags point at the wrong zones

    def test_unresolved_zone_is_flagged_in_brief(self):
        for seed in range(8):
            sc = make_scenario(seed, difficulty=1)
            r = analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                        truth=sc["truth"], settings=Settings(), use_meta=False)
            if any(z["verification"] == "unresolved" for z in r["top"]):
                self.assertIn("VERIFY ON GROUND", r["brief"]["en"]); self.assertIn("মাঠে যাচাই করুন", r["brief"]["bn"]); return
        self.fail("no unresolved top zone found in 8 hard scenes")

    def test_report_and_realeval(self):
        import os, tempfile
        from first72.report import build_html
        from first72.server import build_payload
        from first72.realeval import run as real_run
        sc, r = run(0)
        html = build_html(build_payload(r, sc["pre"], sc["post"], {"synthetic": True}, sc["truth"]))
        self.assertIn("situation report", html); self.assertIn("Audit agent trace", html); self.assertIn("মাঠ প্রতিবেদন", html)
        with tempfile.TemporaryDirectory() as d:
            for name, seed in (("a", 1), ("b", 2)):
                s2 = make_scenario(seed)
                Image.fromarray(s2["post"]).save(f"{d}/{name}_post.png"); Image.fromarray(s2["pre"]).save(f"{d}/{name}_pre.png")
                Image.fromarray((s2["truth"] * 255).astype(np.uint8)).save(f"{d}/{name}_label.png")
            Image.fromarray(sc["post"]).save(f"{d}/c_post.png"); Image.fromarray((sc["truth"] * 255).astype(np.uint8)).save(f"{d}/c_label.png")  # no pre
            rows = real_run(d, out=f"{d}/res.md")
            self.assertEqual(len(rows), 3); self.assertEqual({r_[1] for r_ in rows}, {"change", "post_only"})
            self.assertTrue(os.path.exists(f"{d}/res.md"))


class AgentWithMeta(MetaPaths):
    """Reuses the mock Meta server: Muse chooses the tool order; an invalid tool name must be rejected."""

    def test_meta_planner_used_and_validated(self):
        FakeMeta.mode = "good"
        sc = make_scenario(0, difficulty=1)
        r = analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                    truth=sc["truth"], settings=self.settings(), use_meta=True)
        self.assertEqual(r["telemetry"]["agent"]["planner"], "meta")
        self.assertTrue(any("Planner: Muse" in x for x in r["agent_trace"]))
        self.assertTrue(any("ran local_threshold, alt_index" in x for x in r["agent_trace"]))
        FakeMeta.mode = "bad"  # first zone gets an invalid tool -> that zone falls back to the default plan
        r2 = analyze(sc["pre"], sc["post"], roads=sc["roads"], buildings=sc["buildings"], depot=sc["depot"],
                     truth=sc["truth"], settings=self.settings(), use_meta=True)
        self.assertFalse(any("bogus_tool" in x for x in r2["agent_trace"]))

    # inherited MetaPaths tests are skipped here (they already run once)
    def run(self, result=None):
        if self._testMethodName != "test_meta_planner_used_and_validated":
            return None
        return super().run(result)


if __name__ == "__main__":
    unittest.main()


class OpenWeightsOnlyMode(unittest.TestCase):
    def test_sam_disabled_when_flag_off(self):
        import os
        from first72.config import get_settings
        from first72.sam import MetaSAMRefiner
        old = {k: os.environ.get(k) for k in ("META_USE_SAM", "META_API_BASE", "META_API_KEY")}
        try:
            os.environ.update({"META_USE_SAM": "0", "META_API_BASE": "http://x/v1", "META_API_KEY": "k"})
            s = get_settings()
            self.assertFalse(s.use_sam)
            self.assertFalse(MetaSAMRefiner(s).available())
            os.environ["META_USE_SAM"] = "1"
            self.assertTrue(MetaSAMRefiner(get_settings()).available())
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class CallBudget(unittest.TestCase):
    def test_cap_refuses_after_limit_and_persists(self):
        import os, tempfile
        from first72 import budget
        d = tempfile.mkdtemp()
        old = (budget.PATH, os.environ.get("F72_MAX_API_CALLS"))
        try:
            budget.PATH = os.path.join(d, ".api_calls")
            os.environ["F72_MAX_API_CALLS"] = "5"
            for _ in range(5):
                budget.spend()
            self.assertEqual(budget.used(), 5)
            with self.assertRaises(RuntimeError):
                budget.spend()
            self.assertEqual(budget.used(), 5)  # refused calls are not counted
            budget.reset()
            budget.spend()
            self.assertEqual(budget.used(), 1)
        finally:
            budget.PATH = old[0]
            os.environ.pop("F72_MAX_API_CALLS", None) if old[1] is None else os.environ.__setitem__("F72_MAX_API_CALLS", old[1])


class DotenvParsing(unittest.TestCase):
    def test_inline_comments_are_ignored(self):
        import os, tempfile
        from first72 import config
        d = tempfile.mkdtemp(); p = os.path.join(d, ".env")
        open(p, "w").write('T_A=10   # metres\nT_B=\nT_C="quoted # keep"  # note\nT_D=https://x/v1\n')
        for k in ("T_A", "T_B", "T_C", "T_D"):
            os.environ.pop(k, None)
        config._load_dotenv(p)
        self.assertEqual((os.environ["T_A"], os.environ["T_B"], os.environ["T_C"], os.environ["T_D"]),
                         ("10", "", "quoted # keep", "https://x/v1"))
