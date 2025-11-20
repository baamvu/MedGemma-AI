import os

# Lấy đường dẫn gốc của project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "Models")

# Ưu tiên 1: Model đã Merge (Chạy nhanh nhất, không cần adapter)
MERGED_MODEL_PATH = os.path.join(MODELS_DIR, "medgemma-4b-merged")

# Ưu tiên 2: Base Model + Adapter (Nếu chưa merge)
BASE_MODEL_LOCAL = os.path.join(MODELS_DIR, "base_medgemma4b")
ADAPTER_PATH = os.path.join(MODELS_DIR, "checkpoint-63")

# Fallback: Tải từ HuggingFace nếu không có local
HF_BASE_MODEL = "google/medgemma-4b-it"

def get_model_paths():
    """
    Trả về cấu hình đường dẫn phù hợp nhất dựa trên các file đang có.
    """
    # 1. Kiểm tra Merged Model
    if os.path.exists(os.path.join(MERGED_MODEL_PATH, "config.json")):
        return {
            "type": "merged",
            "base": MERGED_MODEL_PATH,
            "adapter": None,
            "is_local": True
        }
    
    # 2. Kiểm tra Base Local + Adapter
    if os.path.exists(os.path.join(BASE_MODEL_LOCAL, "config.json")):
        adapter = ADAPTER_PATH if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")) else None
        return {
            "type": "base_adapter",
            "base": BASE_MODEL_LOCAL,
            "adapter": adapter,
            "is_local": True
        }

    # 3. Fallback online
    return {
        "type": "online",
        "base": HF_BASE_MODEL,
        "adapter": ADAPTER_PATH if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")) else None,
        "is_local": False
    }

