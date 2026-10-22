from fastapi import APIRouter,HTTPException,UploadFile,File
from pathlib import Path
import pandas as pd,io,json
from ..schemas.prediction import PredictionRequest,PredictionResponse,SequenceRequest,ExplainRequest
from ..ml.predictor import service
from ..ml.data_validation import inspect_dataframe
from ..ml.features import add_past_only_temporal_features
from ..core.config import MAX_UPLOAD_BYTES,REPORT_PATH
from ..services.explain import explain_prediction
router=APIRouter()
@router.get("/health")
def health():return {"status":"ok","model_ready":service.ready}
@router.get("/model-info")
def model_info():return service.info()
@router.get("/evaluation")
def evaluation():
 if not REPORT_PATH.exists():return {"available":False,"message":"No evaluation report. Run training first."}
 return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
@router.post("/predict",response_model=PredictionResponse)
def predict(req:PredictionRequest):
 try:return service.predict_one(req.model_dump())
 except RuntimeError as e:raise HTTPException(503,str(e))
@router.post("/predict-sequence")
def predict_sequence(req:SequenceRequest):
 observations=sorted(req.observations,key=lambda x:x.timestamp)
 stamps=[o.timestamp for o in observations]
 if len(stamps)!=len(set(stamps)):
  raise HTTPException(422,"Observation timestamps must be unique; this prevents ambiguous same-time history features.")
 rows=[]
 for i,obs in enumerate(observations):
  p=obs.model_dump();p["patient_id"]="request-patient";p["prediction_timestamp"]=obs.timestamp;rows.append(p)
 # Feature engineering uses only observations at or before each timestamp, then each row is scored.
 history=add_past_only_temporal_features(pd.DataFrame(rows))
 outputs=[]
 for _,row in history.iterrows():
  p=row.to_dict();p["prediction_timestamp"]=p["timestamp"]
  try:outputs.append(service.predict_one(p))
  except RuntimeError as e:raise HTTPException(503,str(e))
 return {"predictions":outputs,"count":len(outputs),"warning":"Past-only rolling features are computed from this request's timestamped observations. Future rows are not used; this research model is not clinically validated."}
@router.post("/explain")
def explain(req:ExplainRequest):
 if not service.ready:raise HTTPException(503,"No model loaded.")
 return explain_prediction(service.bundle,req.model_dump())
@router.post("/validate-dataset")
async def validate_dataset(file:UploadFile=File(...)):
 if not file.filename or Path(file.filename).suffix.lower()!=".csv":raise HTTPException(400,"Upload a .csv file.")
 content=await file.read(MAX_UPLOAD_BYTES+1)
 if len(content)>MAX_UPLOAD_BYTES:raise HTTPException(413,f"CSV exceeds {MAX_UPLOAD_BYTES} byte limit.")
 try:df=pd.read_csv(io.BytesIO(content))
 except Exception:raise HTTPException(400,"Could not parse CSV.")
 if len(df.columns)>250 or len(df)>1_000_000:raise HTTPException(413,"CSV has too many rows or columns.")
 out=inspect_dataframe(df);out["filename"]=Path(file.filename).name;out["warnings"]=[]
 if "patient_id" not in df:out["warnings"].append("patient_id missing: patient-level split cannot be verified.")
 if not ({"aki_onset_timestamp","aki_label","target_24h"}&set(df.columns)):out["warnings"].append("No obvious outcome/target column found.")
 out["warnings"].append("Inspection only; file is not saved and training is not started.")
 return out
