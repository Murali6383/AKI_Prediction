"""Train/evaluate tabular AKI models with patient-level splits and validation-only tuning."""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from .features import make_features, add_past_only_temporal_features, FEATURE_COLUMNS
from .evaluation import evaluate_probabilities
from ..core.config import MODEL_DIR, REPORT_DIR

def synthetic_dataset(n=1800, seed=42):
    """Clearly synthetic software demo only; not a clinical data substitute."""
    r=np.random.default_rng(seed); age=r.normal(62,16,n).clip(18,95); base=r.lognormal(np.log(.9),.25,n).clip(.4,2.5)
    cr=(base+r.normal(.12,.35,n)+(age>75)*.15).clip(.3,6); bun=r.normal(24,12,n).clip(4,100); egfr=(110-age*.7-cr*12+r.normal(0,10,n)).clip(5,130)
    hr=r.normal(86,15,n).clip(40,160); sbp=r.normal(124,20,n).clip(70,220); dbp=r.normal(74,12,n).clip(35,130)
    na=r.normal(138,4,n).clip(118,155); k=r.normal(4.2,.6,n).clip(2.2,7); bic=r.normal(24,4,n).clip(8,40); urine=r.normal(650,250,n).clip(0,2000)
    sex=r.choice(["female","male","other","unknown"],n,p=[.48,.48,.01,.03]); com=[r.choice(["diabetes","hypertension","heart_failure","none"],size=int(r.integers(0,3)),replace=False).tolist() for _ in range(n)]
    z=-3+.025*(age-60)+1.1*(cr-base)+.018*(bun-20)+.018*(90-egfr)+.004*(100-urine)+.012*(95-sbp)+np.array([.18*len(c) for c in com]); y=r.binomial(1,1/(1+np.exp(-z)))
    d=pd.DataFrame(dict(age=age,sex=sex,creatinine=cr,baseline_creatinine=base,bun=bun,egfr=egfr,heart_rate=hr,systolic_bp=sbp,diastolic_bp=dbp,sodium=na,potassium=k,bicarbonate=bic,urine_output_ml_6h=urine,comorbidities=com,target_24h=y,patient_id=[f"SYN-{i:05d}" for i in range(n)]))
    for c in ["creatinine","bun","egfr","urine_output_ml_6h","sodium"]: d.loc[r.random(n)<.08,c]=np.nan
    return d

def make_pipeline(model):
    nums=[c for c in FEATURE_COLUMNS if c!="sex"]
    prep=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("scale",StandardScaler())]),nums),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),["sex"])])
    return Pipeline([("preprocessor",prep),("model",model)])

def tune_threshold(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    if not len(thresholds): return .5
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-9)
    return float(thresholds[int(np.nanargmax(f))])

def _models():
    models={"Logistic Regression":LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42)}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"]=XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,reg_lambda=2,eval_metric="logloss",random_state=42,n_jobs=4)
    except Exception as e: print("XGBoost unavailable:",e)
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"]=LGBMClassifier(n_estimators=150,num_leaves=15,learning_rate=.04,class_weight="balanced",verbosity=-1,random_state=42,n_jobs=4)
    except Exception as e: print("LightGBM unavailable:",e)
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"]=CatBoostClassifier(iterations=180,depth=5,learning_rate=.04,verbose=False,random_seed=42,allow_writing_files=False,auto_class_weights="Balanced",thread_count=4)
    except Exception as e: print("CatBoost unavailable (optional):",e)
    return models

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--demo",action="store_true"); ap.add_argument("--csv"); ap.add_argument("--target",default="target_24h"); args=ap.parse_args()
    if args.demo: df=synthetic_dataset(); mode="synthetic_demo"
    elif args.csv: df=pd.read_csv(args.csv); mode="research_unvalidated"
    else: ap.error("Choose --demo or --csv PATH")
    if "patient_id" not in df or args.target not in df: raise SystemExit(f"CSV requires patient_id and {args.target}; generate labels first if needed.")
    df=df[df.patient_id.notna() & pd.to_numeric(df[args.target],errors="coerce").isin([0,1])].copy(); df[args.target]=df[args.target].astype(int)
    if "timestamp" in df.columns: df=add_past_only_temporal_features(df)
    # Each patient is assigned wholly to exactly one partition.
    patient_y=df.groupby("patient_id")[args.target].max()
    if patient_y.nunique()<2 or patient_y.value_counts().min()<3: raise SystemExit("Need at least 3 patients in each outcome class for stratified patient-level splitting.")
    tr,tmp=train_test_split(patient_y.index,test_size=.30,random_state=42,stratify=patient_y.values)
    vl,te=train_test_split(tmp,test_size=.50,random_state=42,stratify=patient_y.loc[tmp].values)
    sets=[df[df.patient_id.isin(ids)].copy() for ids in [tr,vl,te]]; xs=[make_features(d) for d in sets]; ys=[d[args.target].to_numpy() for d in sets]
    results={}; pipes={}
    for name,model in _models().items():
        try:
            pipe=make_pipeline(model); pipe.fit(xs[0],ys[0]); pv=pipe.predict_proba(xs[1])[:,1]; threshold=tune_threshold(ys[1],pv)
            results[name]={"validation":evaluate_probabilities(ys[1],pv,threshold),"threshold":threshold}; pipes[name]=pipe
        except Exception as e:
            print(f"Skipping {name}; this installed library combination failed: {type(e).__name__}: {e}")
    if not results: raise SystemExit("No model dependencies available.")
    selected=max(results,key=lambda n:((results[n]["validation"]["auprc"] or 0),(results[n]["validation"]["auroc"] or 0)))
    pt=pipes[selected].predict_proba(xs[2])[:,1]; test=evaluate_probabilities(ys[2],pt,results[selected]["threshold"])
    MODEL_DIR.mkdir(parents=True,exist_ok=True); REPORT_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump({"pipeline":pipes[selected],"model_name":selected,"threshold":results[selected]["threshold"],"dataset_mode":mode,"feature_columns":FEATURE_COLUMNS,"target_column":args.target,"seed":42},MODEL_DIR/"aki_model.joblib")
    report={"dataset_mode":mode,"synthetic_warning":mode=="synthetic_demo","target":args.target,"selection_rule":"Validation AUPRC, then AUROC; threshold tuned on validation F1 only. Test set used once after model selection.","split":{"train_rows":len(sets[0]),"validation_rows":len(sets[1]),"test_rows":len(sets[2]),"train_patients":len(tr),"validation_patients":len(vl),"test_patients":len(te)},"selected_model":selected,"models":results,"test":test,"limitations":["Synthetic demo metrics are not clinical evidence." if mode=="synthetic_demo" else "Retrospective metrics do not imply external or prospective validation.","Risk predictions are research outputs, not medical advice or treatment decisions."]}
    (REPORT_DIR/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"selected_model":selected,"dataset_mode":mode,"test_metrics":test,"report":str(REPORT_DIR/"evaluation.json")},indent=2))
if __name__=="__main__": main()
