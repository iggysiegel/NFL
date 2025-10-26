"""Define project directory structure and ensure necessary folders exist."""

from pathlib import Path

ROOT_DIR = ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREDICTION_DIR = DATA_DIR / "predictions"

for d in [DATA_DIR, RAW_DIR, PREDICTION_DIR]:
    d.mkdir(parents=True, exist_ok=True)
