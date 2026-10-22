import json,joblib,pandas as pd
from datetime import datetime,timezone
from ..core.config import MODEL_PATH,REPORT_PATH
from .features import make_features,FEATURE_COLUMNS
class ModelService:
 def __init__(self): self.bundle=None;self.report={};self.load()
 def load(self):
  if MODEL_PATH.exists():self.bundle=joblib.load(MODEL_PATH)
  if REPORT_PATH.exists():
   try:self.report=json.loads(REPORT_PATH.read_text(encoding="utf-8"))
   except Exception:self.report={}
 @property
 def ready(self):return self.bundle is not None
 def info(self):
  return {"ready":self.ready,"model_name":self.bundle.get("model_name","Not trained") if self.ready else "Not trained","dataset_mode":self.bundle.get("dataset_mode","unavailable") if self.ready else "unavailable","features":FEATURE_COLUMNS,"threshold":self.bundle.get("threshold") if self.ready else None,"warning":"Synthetic demo model; not clinically validated." if self.ready and self.bundle.get("dataset_mode")=="synthetic_demo" else "Research-only; not for treatment decisions."}
 def predict_one(self,payload):
  if not self.ready:raise RuntimeError("No model available. Run python -m backend.app.ml.train --demo")
  x=make_features(pd.DataFrame([payload])); prob=float(self.bundle["pipeline"].predict_proba(x)[0,1]); th=float(self.bundle.get("threshold",.5))
  level="HIGH" if prob>=max(th,.65) else "MODERATE" if prob>=th else "LOW"
  missing=[k for k in ["age","creatinine","baseline_creatinine","bun","egfr"] if payload.get(k) is None]
  warnings=(["Missing potentially relevant fields: "+", ".join(missing)] if missing else [])+["Research prototype only; do not use for clinical decisions."]
  if self.bundle["dataset_mode"]=="synthetic_demo":warnings.insert(0,"Synthetic demo model; not clinically validated.")
  return {"risk_probability":round(prob,6),"risk_level":level,"threshold":th,"model_name":self.bundle["model_name"],"prediction_timestamp":payload.get("prediction_timestamp") or datetime.now(timezone.utc),"dataset_mode":self.bundle["dataset_mode"],"warnings":warnings}
service=ModelService()
