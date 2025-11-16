"""Define directory structure and ensure necessary folders exist."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
PREDICTION_DIR = ROOT_DIR / "predictions"

for d in [SRC_DIR, PREDICTION_DIR]:
    d.mkdir(parents=True, exist_ok=True)
