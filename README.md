# First72

**Disaster imagery to a ranked, uncertainty-aware dispatch plan, in seconds, with an agent that audits its own map.**
Built solo for the Meta Global AI Developer Hackathon (Indie track; themes *Build for Your World* and *Push the Frontier*).

After a flood, the first 72 hours decide who gets help, and responders lose days reading satellite imagery by hand.
First72 takes a **pre/post image pair** and produces:

1. a **flood map** of what is *newly* flooded (not the river that was always there),
2. **per-zone estimates with ranges** (people in flooded buildings, flooded area, blocked roads),
3. a **ranked priority list** with **flood-aware routes**, and zones flagged *boat/air only*,
4. an **audit agent** that cross-checks the zones it is least sure about and flags the ones to verify on the ground,
5. an **English + Bangla field brief** carrying the uncertainty through, plus a printable HTML report.

## Built during the hackathon

Written Oct 27 - Nov 4, 2026 by Md. Monir Hossain Chowdhury (solo, Indie track) with Claude (Anthropic) as an AI coding assistant.
Nothing here pre-dates the hackathon.

## Run it (about a minute, no API key needed)

```bash
./run.sh            # Linux/macOS: venv + deps + tests + web UI  ->  http://127.0.0.1:8072
run.bat             # Windows
# or: pip install -r requirements.txt && python -m first72 serve
```

```bash
python -m first72 demo --seed 2 --difficulty 1     # one synthetic scene -> out/ (overlay, report.html, briefs, json)
python -m first72 benchmark --n 30                 # baseline vs change detection vs +agent -> benchmark/results.md
python -m first72 analyze --pre pre.png --post post.png [--truth mask.png]
python -m first72 evaluate --dir tiles/            # REAL tiles: <name>_post / _label [/ _pre] -> real_results.md
python -m first72 check-meta                       # test your LIVE Meta setup (reasoning model + SAM) -> meta_check.json
python -m unittest discover -s tests -v            # 22 tests
docker build -t first72 . && docker run -p 8072:8072 first72
```

## The audit agent (`first72/agent.py`)

For each uncertain top-priority zone the agent loops: **diagnose** (ambiguous pixels, wide population range) ->
**plan** (Muse picks an ordered tool list; validated against the allowed tools, deterministic default otherwise) ->
**act** (independent evidence tools: `alt_index` second spectral index, `local_threshold` zone-local Otsu,
`sam_reprompt` SAM on the zone crop) -> **verify**. The agent never grades itself by sharpening probabilities; it compares
independent tool outputs:

| Verdict | Meaning | Effect |
|---|---|---|
| `confirmed` | a tool agrees with the baseline mask | estimate unchanged |
| `revised` | two tools agree with each other, not the baseline | adopt their consensus |
| `unresolved` | no agreement | keep baseline, widen the range, flag **verify on ground** (also in the brief) |

Every step is in a trace shown in the UI and the report. A second verifier checks Muse's brief: unknown zone ids, or a first
action outside the top-3 zones, make it reject the answer and fall back to the template (the UI shows why).

## Using the Meta models

Copy `.env.example` to `.env`, fill it in, then run `python -m first72 check-meta` (prints PASS/FAIL with a plain-English hint).

| Stage | Model | Config | Without it |
|---|---|---|---|
| Reasoning, tool planning, bilingual brief | Muse Spark (Meta Model API) **or** Muse Spark Open on Baseten/Modal/Fireworks (`meta-models/muse-spark-open-1026`) | `META_API_BASE`, `META_API_KEY`, `META_REASONING_MODEL` | deterministic EN/BN templates, default tool order |
| Mask refinement + zone re-prompting | SAM 3.1 via Meta Model API (Responses API) | `META_SAM_API_BASE`, `META_SAM_API_KEY`, `META_SAM_MODEL`, `META_SAM_MODE`; `pip install meta-sam-parser` | classical water-index segmentation |

SAM modes: `concept` prompts "flooded land"; `points` builds official object prompts (positive points from high-confidence
change-detection pixels, negative points on confident dry land / permanent water). SAM masks are accepted only where they
agree with the change-detection prior, and SAM can add evidence but never delete it.

**Spend cap:** set `F72_MAX_API_CALLS=5` in `.env` and the app refuses live model calls after 5 (counted in `.api_calls`, survives restarts;
the app falls back to offline mode). `python -m first72 reset-budget` resets it.

### No card for the Meta Model API? Open-weights-only mode

The judging rules accept Meta Model API **and/or** Meta open-weight models. Point `META_API_BASE`/`META_API_KEY` at a Baseten or Modal
endpoint serving Muse Spark Open (`meta-models/muse-spark-open-1026`) and set `META_USE_SAM=0`. Planner, brief and verifier then run on the
open-weights model, segmentation is classical, and `check-meta` reports SAM as skipped on purpose. Do not claim SAM in that mode.

## Honest status (read before the demo)

* **No live Meta call has been made from this code.** Verified: the SAM decoder and object-prompt builder run against Meta's own
  `meta-sam-parser` package and its official sample output, and every Meta code path (chat, tool planner, verifier, SAM
  decode, fallbacks) passes tests against a local mock server. Not verified: the real Model API base URL, model ids, auth
  header and exact request bodies. Run it once with your credits; failures show the exact error and fall back.
  Stage one of judging requires real Meta model use.
* **Benchmark numbers are synthetic**, with exact ground truth. They validate the pipeline; they are not real-world accuracy.
  Use `evaluate` on real tiles and report those numbers, bad tiles included.
* **The default segmenter is an RGB water index.** It fails on muddy floodwater (see below). For real Sentinel-2 swap in
  NDWI/MNDWI in `segmentation.water_index`.
* Urgency weights, needs rules and travel times are transparent heuristics, not validated humanitarian doctrine. This is
  **decision support**: verify on the ground before committing assets. No medical advice is generated.

## Synthetic benchmark (30 scenes per difficulty; full tables in `benchmark/results.md`)

| Scenes | Method | IoU | Precision | Recall | ECE | Top-5 zone overlap |
|---|---|---|---|---|---|---|
| Easy | Post-only threshold (baseline) | 0.831 | 0.833 | 0.996 | 0.071 | 0.887 |
| Easy | Change detection | 0.987 | 0.995 | 0.992 | 0.041 | 0.933 |
| Easy | Change detection + agent | 0.987 | 0.995 | 0.992 | 0.041 | 0.933 |
| Hard (muddy water, haze) | Post-only threshold | 0.640 | 0.643 | 0.993 | 0.172 | 0.773 |
| Hard | Change detection | 0.587 | 0.951 | 0.604 | 0.081 | 0.540 |
| Hard | Change detection + agent | 0.590 | 0.951 | 0.607 | 0.080 | 0.527 |

What this does and does not show:

* On easy scenes, change detection is far better than the naive baseline (0.987 vs 0.831 IoU). The agent confirms everything and changes nothing.
* On hard scenes **every method is weak**: muddy floodwater is not water-coloured, so the index misses it (recall 0.60). The post-only
  baseline scores a higher IoU there (0.640) because it over-detects. **The agent does not fix this accuracy gap.**
* What the agent *does* provide is triage. Mean zone IoU of audited zones, hard scenes: `confirmed` 0.830, `revised` 0.924,
  `unresolved` 0.462. The "verify on ground" flags land on the zones that really are wrong.
* An earlier version averaged tool outputs into the map and *hurt* ranking; a confirmation now never changes the estimate.
  A tried `change_magnitude` tool gave no measurable gain and was removed.

## Layout

```
first72/imagery.py       synthetic pre/post scenes (easy/hard) + ground truth
first72/segmentation.py  change-detection flood probability, alt index, calibration
first72/sam.py           SAM 3.1 adapter (Responses API + meta-sam-parser; concept & points; safe fallback)
first72/agent.py         audit agent: diagnose, plan, act, verify, trace
first72/decision.py      zone stats with lo/exp/hi ranges, urgency ranking, Dijkstra routing
first72/reasoner.py      Meta chat client + brief verifier + tool planner + EN/BN offline templates
first72/pipeline.py      orchestration     first72/metrics.py  IoU/PRF1, ECE, top-k, coverage
first72/realeval.py      evaluate on real tiles     first72/report.py  printable HTML report
first72/server.py + web/ stdlib server and UI       docs/  submission guide + demo script
```

## Submitting

See `docs/REQUIREMENTS_CHECK.md` (what is done vs yours), `docs/SUBMISSION_GUIDE.md` (step by step, deadline in Dhaka time),
`docs/PROJECT_SUMMARY.md` (paste-ready summary) and `docs/DEMO_SCRIPT.md` (3-minute video script).

Data: if you use Sentinel-2 or Sen1Floods11, cite their licences here. License: MIT
