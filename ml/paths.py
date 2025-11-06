import os
from pathlib import Path

APP_ROOT = Path(os.getenv("APP_ROOT", "/app")).resolve()

MODEL_DIR = Path(os.getenv("MODEL_DIR", APP_ROOT / "model")).resolve()

DATA_DIR = Path(os.getenv("DATA_DIR", APP_ROOT / "data")).resolve()

TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp")).resolve()

CONFIGS_DIR = Path(os.getenv("CONFIGS", APP_ROOT / "configs")).resolve()


# Would be better to use Artifact for the name, but will keep it as model for the assignment
#ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", APP_ROOT / "artifacts")).resolve()

# Direct File ASsignment, I won't be using these

#MODEL_FILE = ARTIFACT_DIR / "model.pkl"
#ENCODER_FILE = ARTIFACT_DIR / "encoder.pkl"
#LB_FILE = ARTIFACT_DIR / "label_binarizer.pkl"
#FEATURES_FILE = ARTIFACT_DIR / "features.json"