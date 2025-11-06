from pathlib import Path
import yaml
import re


from ml.paths import APP_ROOT, CONFIGS_DIR

def _load_yaml(path: str | Path):
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

        

def load_config(config_path: str | Path):
    """
    Load a model-specific config and shallow-merge it with base.yaml if present.
    Sections train/model are merged key-by-key.
    """
    #config_path = Path(config_path).resolve()
    config_path = CONFIGS_DIR
    cfg = _load_yaml(config_path)

    base_path = cfg.get("_base")
    if base_path:
        base_path = (config_path.parent / base_path).resolve()

        base = _load_yaml(base_path)
        merged = {**base, **cfg}  # top-level shallow merge
        
        for section in ("train", "model"):
            merged[section] = {**base.get(section, {}), **cfg.get(section, {})}
        return merged

    return cfg



def get_model_name_from_cfg(cfg):
    """Return a filesystem-safe model name derived from class_path in config."""
    
    model_cfg = cfg.get("model", {})
    class_path = model_cfg.get("class_path", "unknown_model")
    base = class_path.split(".")[-1]  # e.g. RandomForestClassifier
    
    return re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()