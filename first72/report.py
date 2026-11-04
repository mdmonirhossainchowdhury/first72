"""Standalone, printable HTML situation report (images embedded; opens offline, prints to PDF)."""
import html

NEED = {"water": "water", "shelter": "shelter", "food": "food", "medical_triage": "medical triage", "boats": "boats/air"}
VER = {"confirmed": "confirmed by independent check", "revised": "revised after cross-check",
       "unresolved": "VERIFY ON GROUND", "not_audited": "-"}


def build_html(p):
    e = html.escape
    t = p["totals"]
    rows = "".join(
        f"<tr><td>{z['rank']}</td><td><b>{z['id']}</b></td><td>{round(z['population']['exp'])} "
        f"({round(z['population']['lo'])}-{round(z['population']['hi'])})</td><td>{round(z['flooded_frac'] * 100)}%</td>"
        f"<td>{'road ~' + str(z['eta_min']) + ' min' if z['access'] == 'road' else 'boat / air'}</td>"
        f"<td>{e(', '.join(NEED[n] for n in z['needs']))}</td><td>{e(VER.get(z.get('verification', 'not_audited'), ''))}</td></tr>"
        for z in p["top"])
    trace = "".join(f"<li>{e(x)}</li>" for x in p.get("agent_trace", []))
    m = p.get("metrics")
    met = ""
    if m:
        met = ("<h2>Evaluation vs ground truth</h2><p>" + e(" | ".join(f"{k}: {v:.3f}" for k, v in m.items())) + "</p>"
               + ("<p class='w'>Synthetic scene: not real-world accuracy.</p>" if p.get("scene", {}).get("synthetic") else ""))
    tel = p.get("telemetry", {})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>First72 situation report</title>
<style>body{{font:14px/1.5 system-ui,'Noto Sans Bengali',sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#111}}
h1 b{{color:#e0245e}} table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #ddd;padding:5px;text-align:left}}
pre{{white-space:pre-wrap;background:#f6f7f9;padding:10px;border-radius:8px}} img{{width:100%;border-radius:8px}} .w{{color:#b45f00}}
@media print{{body{{margin:0}}}}</style></head><body>
<h1><b>First</b>72 &mdash; situation report</h1>
<p>Newly flooded: <b>{t['flooded_km2']:.1f} km&sup2;</b> &middot; people in flooded buildings: <b>~{round(t['people_exp'])}</b>
(plausible {round(t['people_lo'])}-{round(t['people_hi'])}) &middot; zones affected: {t['zones_affected']}</p>
<img src="{p['layers']['overlay']}" alt="analysis map">
<h2>Priority zones</h2><table><tr><th>#</th><th>Zone</th><th>People (range)</th><th>Flooded</th><th>Access</th><th>Send</th><th>Verification</th></tr>{rows}</table>
<h2>Field brief (English)</h2><pre>{e(p['brief']['en'])}</pre>
<h2>মাঠ প্রতিবেদন (বাংলা)</h2><pre>{e(p['brief']['bn'])}</pre>
<h2>Audit agent trace</h2><ul>{trace or '<li>No audit needed.</li>'}</ul>
{met}
<p style="color:#666;font-size:12px">Brief source: {e(p['brief']['source'])} &middot; vision {tel.get('vision_seconds', '?')}s,
reasoning {tel.get('reasoning_seconds', '?')}s. Automated decision support: verify on the ground before committing scarce assets.</p>
</body></html>"""
