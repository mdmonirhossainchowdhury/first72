"""Optional SAM 3.1 refinement through the Meta Model API (Responses API + Meta's `meta-sam-parser`).

Protocol (from Meta's `meta-sam-parser` package docs): POST a Responses-API request whose input holds an
`input_image` plus an `input_text`; SAM 3.1 answers with special-token text that `meta_sam_parser` decodes
into box + mask records. The decoder here is tested against the package's own sample output. What is NOT
verified: the exact Model API URL / model id / auth header (take them from Meta's docs and put them in .env)
and the live request body. Any failure is recorded and the classical result is kept, so the app never breaks.

Two prompting modes (META_SAM_MODE):
  concept (default)  the noun phrase "flooded land" -> every matching instance
  points             official object prompts (`build_object_prompt`): each object gets positive points
                     sampled from high-confidence change-detection flood pixels plus negative points on
                     confident dry land / permanent water

SAM masks are kept only where they agree with the pre/post change-detection prior, so SAM contributes
sharp boundaries while change detection decides *what is newly flooded*.

Install the decoder:  pip install meta-sam-parser   (or github.com/meta-models/meta-sam)
"""
import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image

CONCEPT = "flooded land"


def _png_data_url(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _post_json(url, key, body, timeout):
    from . import budget
    budget.spend()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def responses_text(resp):
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    parts = []
    for item in resp.get("output", []) or []:
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                parts.append(c.get("text", ""))
    return "".join(parts)


def parse_sam_text(text, size):
    """Decode SAM 3.1 special-token text -> [(full_frame_bool_mask, confidence)]. size = (width, height)."""
    from meta_sam_parser import (CompletedOutcome, SegmentationMaskRecord, decode_mask_to_raster,
                                 image_segmentation_format)
    w, h = size
    parser = image_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    out = []
    for rec in result.records:
        if not isinstance(rec, SegmentationMaskRecord):
            continue
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(rec.mask.height, rec.mask.width) > 0
        b = rec.bounds
        left, top, right, bottom = (int(round(v)) for v in (b.left, b.top, b.right, b.bottom))
        piece = Image.fromarray((raster * 255).astype(np.uint8)).resize(
            (max(1, right - left), max(1, bottom - top)), Image.NEAREST)
        canvas = Image.new("L", (w, h), 0)
        canvas.paste(piece, (left, top))
        out.append((np.array(canvas) > 127, 1.0 if rec.confidence is None else float(rec.confidence)))
    return out


class MetaSAMRefiner:
    def __init__(self, settings):
        self.s = settings
        self.mode = settings.sam_mode

    def available(self):
        return bool(self.s.sam_api_key and (self.s.sam_endpoint or self.s.sam_api_base))

    def _input_text(self, prob, perm, size):
        if self.mode != "points":
            return CONCEPT
        from meta_sam_parser import FramePrompt, PromptObject, PromptPoint, build_object_prompt
        rng = np.random.default_rng(0)
        pos = np.argwhere(prob > 0.9)
        neg = np.argwhere((prob < 0.05) | (perm > 0.9))
        if not len(pos):
            return CONCEPT
        objs = []
        for i, pi in enumerate(rng.choice(len(pos), size=min(6, len(pos)), replace=False)):
            pts = [PromptPoint(int(pos[pi][1]), int(pos[pi][0]))]
            if len(neg):
                for ni in rng.choice(len(neg), size=min(2, len(neg)), replace=False):
                    pts.append(PromptPoint(int(neg[ni][1]), int(neg[ni][0]), label="negative"))
            objs.append(PromptObject(id=i + 1, prompts=[FramePrompt(frame=0, points=pts)]))
        return build_object_prompt(size=size, objects=objs)

    def refine(self, post, prob, perm):
        info = {"used": False, "error": None, "mode": self.mode}
        try:
            h, w = prob.shape
            url = self.s.sam_endpoint or f"{self.s.sam_api_base}/responses"
            body = {"model": self.s.sam_model, "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": _png_data_url(post)},
                {"type": "input_text", "text": self._input_text(prob, perm, (w, h))}]}]}
            data = _post_json(url, self.s.sam_api_key, body, self.s.timeout_s)
            masks = parse_sam_text(responses_text(data), (w, h))
            if not masks:
                info["error"] = "SAM returned no masks"
                return prob, info
            union = np.zeros(prob.shape, bool)
            for m, conf in masks:
                if conf >= 0.5 and m.sum() and (m & (prob > 0.5)).sum() >= 0.5 * m.sum():
                    union |= m  # keep masks that mostly agree with the change-detection prior
            # SAM adds evidence where it agrees; it never deletes change-detection evidence
            refined = (prob + 0.4 * union.astype(np.float32) * (1 - prob)) * (1 - perm)
            info.update(used=True, masks=len(masks), kept_px=int(union.sum()))
            return refined.astype(np.float32), info
        except Exception as e:  # noqa: BLE001 - deliberate: never break the pipeline
            info["error"] = f"{type(e).__name__}: {e}"
            return prob, info
