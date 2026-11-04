"""Reasoning layer.

* MetaReasoner  - calls a Meta model (Muse Spark via Meta Model API, or Muse Spark Open on
                  Baseten/Modal/Fireworks) through an OpenAI-compatible /chat/completions endpoint.
* OfflineReasoner - deterministic templates (EN + Bangla). Always available; also the fallback.
* A VERIFIER checks every model answer against the computed data before it is shown. If the
  model cites an unknown zone, or ranks a non-top-3 zone first, we discard it and fall back.
"""
import json
import re
import urllib.request

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
NEED_EN = {"water": "drinking water", "shelter": "emergency shelter", "food": "food packs",
           "medical_triage": "medical triage team", "boats": "boats / air access"}
NEED_BN = {"water": "পানীয় জল", "shelter": "জরুরি আশ্রয়", "food": "খাদ্য সহায়তা",
           "medical_triage": "চিকিৎসা দল", "boats": "নৌকা বা আকাশপথ"}


def bn(x):
    return str(x).translate(BN_DIGITS)


def _fmt_range(p):
    return f"{round(p['exp'], -1):.0f} ({round(p['lo'], -1):.0f}-{round(p['hi'], -1):.0f})"


def build_summary(zones, top, routes_info, totals):
    return {
        "totals": totals,
        "priority_zones": [{
            "zone": z["id"], "rank": z["rank"], "urgency": round(z["urgency"], 1),
            "people_in_flooded_buildings": {k: round(v) for k, v in z["population"].items()},
            "flooded_percent": round(100 * z["flooded_frac"]),
            "flooded_km2": round(z["area_km2"]["exp"], 2),
            "roads_blocked_percent": round(100 * z["road_blocked_frac"]),
            "access": z["access"], "eta_min": z["eta_min"], "needs": z["needs"],
            "certainty_percent": round(100 * z["certainty"]),
            "verification": z.get("verification", "not_audited"),
        } for z in top],
    }


class OfflineReasoner:
    source = "offline-template"
    calls = 0

    def reason(self, summary):
        t, zs = summary["totals"], summary["priority_zones"]
        en = [f"FIRST72 FIELD BRIEF - about {t['flooded_km2']:.1f} km2 newly flooded; "
              f"roughly {t['people_exp']:.0f} people live in flooded buildings "
              f"(plausible range {t['people_lo']:.0f}-{t['people_hi']:.0f}).", "Priority zones:"]
        bnl = [f"ফার্স্ট৭২ মাঠ প্রতিবেদন - প্রায় {bn(round(t['flooded_km2'], 1))} বর্গকিমি নতুন প্লাবিত; "
               f"প্লাবিত ভবনে আনুমানিক {bn(round(t['people_exp']))} জন মানুষ "
               f"(সম্ভাব্য সীমা {bn(round(t['people_lo']))}-{bn(round(t['people_hi']))})।", "অগ্রাধিকার এলাকা:"]
        actions = []
        for z in zs:
            p = z["people_in_flooded_buildings"]
            access_en = (f"road, ~{z['eta_min']} min" if z["access"] == "road" else "NOT reachable by road")
            access_bn = (f"সড়কপথে, প্রায় {bn(z['eta_min'])} মিনিট" if z["access"] == "road"
                         else "সড়কপথে পৌঁছানো যাবে না")
            needs_en = ", ".join(NEED_EN[n] for n in z["needs"]) or "monitor"
            needs_bn = ", ".join(NEED_BN[n] for n in z["needs"]) or "পর্যবেক্ষণ"
            flag_en = " [VERIFY ON GROUND: independent checks disagreed]" if z.get("verification") == "unresolved" else ""
            flag_bn = " [মাঠে যাচাই করুন]" if z.get("verification") == "unresolved" else ""
            en.append(f"{z['rank']}. Zone {z['zone']}: ~{p['exp']} people ({p['lo']}-{p['hi']}), "
                      f"{z['flooded_percent']}% flooded, access: {access_en}. Send: {needs_en}.{flag_en}")
            bnl.append(f"{bn(z['rank'])}। এলাকা {z['zone']}: আনুমানিক {bn(p['exp'])} জন "
                       f"({bn(p['lo'])}-{bn(p['hi'])}), {bn(z['flooded_percent'])}% প্লাবিত, "
                       f"প্রবেশ: {access_bn}। প্রয়োজন: {needs_bn}।{flag_bn}")
            actions.append({"zone": z["zone"], "action": f"Dispatch: {needs_en}",
                            "why": f"{p['exp']} people at risk, {z['flooded_percent']}% flooded"})
        en.append("Caveats: automated estimate from imagery; ranges show model uncertainty. "
                  "Verify on the ground before committing scarce assets.")
        bnl.append("সতর্কতা: এটি ছবি থেকে স্বয়ংক্রিয় অনুমান; সীমা অনিশ্চয়তা নির্দেশ করে। "
                   "সীমিত সম্পদ পাঠানোর আগে মাঠপর্যায়ে যাচাই করুন।")
        return {"en": "\n".join(en), "bn": "\n".join(bnl), "actions": actions,
                "source": self.source, "verified": True, "notes": []}


SYSTEM_PROMPT = (
    "You are a decision-support assistant for disaster responders. You receive JSON computed by a "
    "vision pipeline. Use ONLY numbers and zone ids present in the JSON; never invent data. "
    "Respond with a single JSON object with keys: en_brief (string), bn_brief (string, Bangla), "
    "actions (list of {zone, action, why}). Keep each brief under 180 words, mention uncertainty "
    "ranges, flag zones reachable only by boat/air, and do not give medical advice."
)


TOOLS = ("alt_index", "local_threshold", "sam_reprompt")
TOOL_PLANNER_PROMPT = (
    "You are a tool planner for a flood-mapping audit agent. For each zone, choose an ordered list of 1-3 tools "
    "from: alt_index (second spectral index; good when lighting/haze is suspected), local_threshold (zone-local "
    "threshold; good for shadows or mixed backgrounds), sam_reprompt (SAM on the zone crop; good for ragged "
    "boundaries). Use only the diagnostics given. Return a single JSON object: {\"plans\": {\"<zone>\": [tools]}}.")


class MetaReasoner:
    def __init__(self, settings):
        self.s = settings
        self.source = f"meta:{settings.reasoning_model}"
        self.fallback = OfflineReasoner()

    @staticmethod
    def available(settings):
        return bool(settings.api_base and settings.api_key)

    calls = 0

    def _chat(self, summary, system=SYSTEM_PROMPT):
        self.calls += 1
        body = json.dumps({
            "model": self.s.reasoning_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(summary, ensure_ascii=False)}],
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{self.s.api_base}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.s.api_key}"})
        from . import budget
        budget.spend()
        with urllib.request.urlopen(req, timeout=self.s.timeout_s) as r:
            data = json.loads(r.read())
        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0))

    def plan_tools(self, diags):
        """Muse chooses which evidence tools to run per uncertain zone. Output is validated; {} -> default plan."""
        try:
            ans = self._chat({"zones": diags}, TOOL_PLANNER_PROMPT)
            plans = ans.get("plans", {})
            ids = {d["zone"] for d in diags}
            ok = {}
            for z, tools in plans.items():
                if z in ids and isinstance(tools, list) and tools and all(x in TOOLS for x in tools):
                    ok[z] = list(dict.fromkeys(tools))
            return ok
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def verify(answer, summary):
        """Return list of problems; empty list == verified."""
        problems = []
        valid = {z["zone"] for z in summary["priority_zones"]}
        top3 = [z["zone"] for z in summary["priority_zones"][:3]]
        for k in ("en_brief", "bn_brief"):
            if not isinstance(answer.get(k), str) or len(answer[k]) < 20:
                problems.append(f"missing/short {k}")
        mentioned = set(re.findall(r"\b[A-H][1-8]\b", (answer.get("en_brief") or "") + " " + (answer.get("bn_brief") or "")))
        unknown = mentioned - valid
        if unknown:
            problems.append(f"cites unknown zones {sorted(unknown)}")
        acts = answer.get("actions") or []
        if acts and acts[0].get("zone") not in top3:
            problems.append("first action is not a top-3 zone")
        if any(a.get("zone") not in valid for a in acts):
            problems.append("action cites unknown zone")
        return problems

    def reason(self, summary):
        offline = self.fallback.reason(summary)
        try:
            ans = self._chat(summary)
            problems = self.verify(ans, summary)
            if problems:
                offline["notes"] = [f"Meta answer rejected by verifier: {'; '.join(problems)}. Used offline template."]
                return offline
            return {"en": ans["en_brief"], "bn": ans["bn_brief"], "actions": ans.get("actions", []),
                    "source": self.source, "verified": True, "notes": []}
        except Exception as e:  # noqa: BLE001
            offline["notes"] = [f"Meta call failed ({type(e).__name__}: {e}). Used offline template."]
            return offline


def get_reasoner(settings, use_meta=True):
    if use_meta and MetaReasoner.available(settings):
        return MetaReasoner(settings)
    return OfflineReasoner()
