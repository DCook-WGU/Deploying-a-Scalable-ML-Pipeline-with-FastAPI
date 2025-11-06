import os
from pathlib import Path

APP_ROOT = Path(os.getenv("APP_ROOT", "/app")).resolve()

MODEL_DIR = Path(os.getenv("MODEL_DIR", APP_ROOT / "model")).resolve()

DATA_DIR = Path(os.getenv("DATA_DIR", APP_ROOT / "data")).resolve()

TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp")).resolve()

CONFIGS_DIR = Path(os.getenv("CONFIGS", APP_ROOT / "configs")).resolve()


