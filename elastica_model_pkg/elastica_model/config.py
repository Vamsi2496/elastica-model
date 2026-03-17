import os
import json

# ── Model constants (el3-3.f90 + c-2.el3) ────────────────────
NDIM     = 4
NPAR     = 9
PARNAMES = {
    1: "Fx", 2: "Fy", 3: "x",  4: "y",
    5: "Asymmetric", 6: "Symmetric",
    7: "MA", 8: "MB", 9: "PE"
}
UNAMES = {1: "theta", 2: "thetaprime", 3: "x_s", 4: "y_s"}

# ── User paths ────────────────────────────────────────────────
# Priority: env var → config file → fallback default

CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), ".elastica_model", "config.json"
)

def _load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

_cfg = _load_config()

PYTHON26 = (
    os.environ.get("ELASTICA_PYTHON26")
    or _cfg.get("PYTHON26")
    or r"C:\Python26\python.exe"
)

AUTO_DIR = (
    os.environ.get("ELASTICA_AUTO_DIR")
    or _cfg.get("AUTO_DIR")
    or r"C:\auto07p\python"
)
