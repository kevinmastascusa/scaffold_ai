"""
Model version/hash logging utility for Scaffold AI.
Logs model name, version, and (if available) model card hash to a log file or test report.
"""
import os
import json
from datetime import datetime
from huggingface_hub import HfApi
from scaffold_core.config import MODEL_REGISTRY

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "model_version_log.json")

def get_model_card_hash(model_id):
    try:
        api = HfApi()
        card = api.model_info(model_id)
        return card.sha
    except Exception:
        return None

def log_model_versions():
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "models": {}
    }
    for model_type, registry in MODEL_REGISTRY.items():
        for key, info in registry.items():
            model_id = info["name"]
            hash_val = get_model_card_hash(model_id)
            log["models"][model_id] = {
                "type": model_type,
                "desc": info.get("desc", ""),
                "hash": hash_val
            }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
    print(f"Model versions logged to {LOG_PATH}")

if __name__ == "__main__":
    log_model_versions()
