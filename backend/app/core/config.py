import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR, REPORT_DIR, DATA_DIR = ROOT/"models", ROOT/"reports", ROOT/"data"
MODEL_PATH, REPORT_PATH = MODEL_DIR/"aki_model.joblib", REPORT_DIR/"evaluation.json"
MAX_UPLOAD_BYTES = int(os.getenv("AKI_MAX_UPLOAD_BYTES", 8*1024*1024))
FRONTEND_ORIGINS = [x.strip() for x in os.getenv("AKI_FRONTEND_ORIGINS","http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
