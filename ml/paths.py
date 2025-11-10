from pathlib import Path


# Establish APP Root
APP_ROOT = Path(__file__).resolve().parents[1]


# Main Folder
DATA_DIR = APP_ROOT / "data"

MODELS_DIR = APP_ROOT / "models"

CONFIGS_DIR = APP_ROOT / "configs"

LOGS_DIR = APP_ROOT / "logs"

SCREENSHOTS_DIR = APP_ROOT / "screenshots"

DOCS_DIR = APP_ROOT / "docs"


# Setup Model Directory Defaults

DEFAULT_MODEL_SUBDIR = "random_forest"
DEFAULT_MODEL_DIR = MODELS_DIR / DEFAULT_MODEL_SUBDIR


def ensure_directories() -> None:
    """Create all common directories if missing."""

    for directory in [
        DATA_DIR,
        MODELS_DIR,
        CONFIGS_DIR,
        LOGS_DIR,
        SCREENSHOTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


# Runs once on import to verify all directories are there and created.
ensure_directories()


# Debug by running directly
if __name__ == "__main__":
    print(f"Project root: {APP_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Configs directory: {CONFIGS_DIR}")
