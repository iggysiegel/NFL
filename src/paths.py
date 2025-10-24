"""Define project directory structure and ensure necessary folders exist."""

from pathlib import Path

ROOT_DIR = ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESS_DIR = DATA_DIR / "process"
FEATURE_DIR = DATA_DIR / "feature"

for d in [DATA_DIR, RAW_DIR, PROCESS_DIR, FEATURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
