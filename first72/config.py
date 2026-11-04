"""Runtime settings, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if v.startswith("#"):
                v = ""                              # KEY=   # note  (empty value)
            elif v[:1] not in ("'", '"') and " #" in v:
                v = v.split(" #", 1)[0].strip()   # inline comment: KEY=value   # note
            elif v[:1] in ("'", '"') and v.count(v[0]) >= 2:
                v = v[1:v.index(v[0], 1)]
            os.environ.setdefault(k.strip(), v)


@dataclass
class Settings:
    # --- Meta Model API / open-weights endpoint (OpenAI-compatible chat completions) ---
    api_base: str = ""
    api_key: str = ""
    reasoning_model: str = "meta-models/muse-spark-open-1026"
    # --- SAM 3.1 via Meta Model API (request schema must be confirmed against the docs) ---
    sam_api_base: str = ""         # Meta Model API base (defaults to META_API_BASE); SAM is called at {base}/responses
    sam_api_key: str = ""          # defaults to META_API_KEY
    sam_endpoint: str = ""         # optional full-URL override
    sam_model: str = "sam-3.1"
    sam_mode: str = "concept"      # "concept" ("flooded land") or "points" (object prompts); see sam.py
    use_sam: bool = True           # META_USE_SAM=0 -> open-weights-only mode (reasoning via partner endpoint, no SAM)
    # --- geometry / analysis ---
    pixel_m: float = 10.0          # ground sampling distance, metres per pixel (Sentinel-2 = 10)
    grid: int = 8                  # zones per side
    default_density: float = 1500  # people per km^2 when no building footprints are supplied
    road_speed_kmh: float = 30.0
    timeout_s: float = 60.0


def get_settings() -> Settings:
    _load_dotenv()
    e = os.environ.get
    use_sam = e("META_USE_SAM", "1").strip().lower() not in ("0", "false", "no", "off")
    return Settings(
        use_sam=use_sam,
        api_base=e("META_API_BASE", "").rstrip("/"),
        api_key=e("META_API_KEY", ""),
        reasoning_model=e("META_REASONING_MODEL", Settings.reasoning_model),
        sam_api_base=((e("META_SAM_API_BASE") or e("META_API_BASE", "")) if use_sam else "").rstrip("/"),
        sam_api_key=(e("META_SAM_API_KEY") or e("META_API_KEY", "")) if use_sam else "",
        sam_endpoint=e("META_SAM_ENDPOINT", "") if use_sam else "",
        sam_model=e("META_SAM_MODEL", Settings.sam_model),
        sam_mode=e("META_SAM_MODE", Settings.sam_mode),
        pixel_m=float(e("F72_PIXEL_METERS", Settings.pixel_m)),
        grid=int(e("F72_GRID", Settings.grid)),
        default_density=float(e("F72_DEFAULT_DENSITY", Settings.default_density)),
        road_speed_kmh=float(e("F72_ROAD_SPEED_KMH", Settings.road_speed_kmh)),
        timeout_s=float(e("F72_TIMEOUT_S", Settings.timeout_s)),
    )
