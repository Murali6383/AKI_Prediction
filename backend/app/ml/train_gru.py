"""Optional GRU sequence benchmark. Requires timestamped repeated observations and PyTorch.
CSV must contain patient_id,timestamp,target_24h plus the same clinical features used by train.py.
Each training example is a prefix ending at its own prediction timestamp; future rows are never included.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, brier_score_loss
from .features import add_past_only_temporal_features, make_features, FEATURE_COLUMNS
from ..core.config import MODEL_DIR, REPORT_DIR

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--csv",required=True); ap.add_argument("--target",default="target_24h"); ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--max-steps",type=int,default=12); args=ap.parse_args()
    try:
        import torch
        from torch import nn
    except ImportError as e: raise SystemExit("PyTorch is optional and not installed. Install a compatible torch build before running GRU.") from e
    torch.manual_seed(42); np.random.seed(42)
    df=pd.read_csv(args.csv)
    if not {"patient_id","timestamp",args.target}.issubset(df): raise SystemExit("CSV must include patient_id,timestamp and the requested target.")
    df["timestamp"]=pd.to_datetime(df.timestamp,errors="coerce",utc=True)
    df=df.loc[df.patient_id.notna()&df.timestamp.notna()&pd.to_numeric(df[args.target],errors="coerce").isin([0,1])].copy()
    df[args.target]=df[args.target].astype(int); df=add_past_only_temporal_features(df)
    patient_y=df.groupby("patient_id")[args.target].max()
    if patient_y.nunique()<2 or patient_y.value_counts().min()<3: raise SystemExit("Need at least 3 patients in each class.")
    train_ids, tmp=train_test_split(patient_y.index,test_size=.30,random_state=42,stratify=patient_y.values)
    val_ids,test_ids=train_test_split(tmp,test_size=.50,random_state=42,stratify=patient_y.loc[tmp].values)
    train_rows=df[df.patient_id.isin(train_ids)].copy(); val_rows=df[df.patient_id.isin(val_ids)].copy(); test_rows=df[df.patient_id.isin(test_ids)].copy()
    # Fit preprocessing on training rows only.
    train_x=make_features(train_rows); numeric=[c for c in FEATURE_COLUMNS if c!="sex"]
    imputer=SimpleImputer(strategy="median",add_indicator=False,keep_empty_features=True); scaler=StandardScaler()
    a=imputer.fit_transform(train_x[numeric]); a=scaler.fit_transform(a)
    sex_levels={v:i for i,v in enumerate(sorted(train_x.sex.fillna("unknown").astype(str).unique()))}
    def matrix(frame):
        f=make_features(frame); n=scaler.transform(imputer.transform(f[numeric])); sex=f.sex.fillna("unknown").astype(str).map(sex_levels).fillna(-1).to_numpy(dtype=float)[:,None]
        return np.concatenate([n,sex],axis=1).astype(np.float32)
    # Build each row's sequence from same patient's past/current rows only.
    def examples(rows):
        mat=matrix(rows); pos={idx:i for i,idx in enumerate(rows.index)}; X=[]; y=[]
        for _,g in rows.sort_values(["patient_id","timestamp"]).groupby("patient_id",sort=False):
            ids=[pos[i] for i in g.index]
            for j, rid in enumerate(ids):
                seq=mat[ids[max(0,j-args.max_steps+1):j+1]]
                padded=np.zeros((args.max_steps,mat.shape[1]),dtype=np.float32); padded[-len(seq):]=seq
                X.append(padded); y.append(int(rows.loc[g.index[j],args.target]))
        return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.float32)
    Xtr,ytr=examples(train_rows); Xv,yv=examples(val_rows); Xte,yte=examples(test_rows)
    class GRUModel(nn.Module):
        def __init__(self,n_features):
            super().__init__(); self.gru=nn.GRU(n_features,32,batch_first=True); self.drop=nn.Dropout(.2); self.fc=nn.Linear(32,1)
        def forward(self,x):
            out,_=self.gru(x); return self.fc(self.drop(out[:,-1,:])).squeeze(-1)
    model=GRUModel(Xtr.shape[-1]); opt=torch.optim.Adam(model.parameters(),lr=.001); loss_fn=nn.BCEWithLogitsLoss()
    xt=torch.tensor(Xtr); yt=torch.tensor(ytr); xv=torch.tensor(Xv)
    best=None; best_ap=-1; best_state=None; patience=4; stale=0
    for epoch in range(args.epochs):
        model.train(); order=torch.randperm(len(xt))
        for start in range(0,len(order),64):
            ids=order[start:start+64]; opt.zero_grad(); loss=loss_fn(model(xt[ids]),yt[ids]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pv=torch.sigmoid(model(xv)).numpy()
        apv=float(average_precision_score(yv,pv)) if len(np.unique(yv))>1 else 0.0
        print(f"epoch={epoch+1} validation_auprc={apv:.4f}")
        if apv>best_ap: best_ap=apv; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pv=torch.sigmoid(model(xv)).numpy(); pt=torch.sigmoid(model(torch.tensor(Xte))).numpy()
    # Threshold chosen on validation only; test evaluated after freezing threshold.
    thresholds=np.unique(np.r_[0.01,pv,0.99]); threshold=max(thresholds,key=lambda t:f1_score(yv,(pv>=t).astype(int),zero_division=0))
    def metrics(y,p):
        pred=(p>=threshold).astype(int)
        return {"auroc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,"auprc":float(average_precision_score(y,p)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"brier_score":float(brier_score_loss(y,p)),"threshold":float(threshold)}
    MODEL_DIR.mkdir(parents=True,exist_ok=True); REPORT_DIR.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"n_features":Xtr.shape[-1],"max_steps":args.max_steps,"feature_columns":FEATURE_COLUMNS,"numeric_columns":numeric,"sex_levels":sex_levels,"imputer_statistics":imputer.statistics_.tolist(),"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),"threshold":float(threshold),"target":args.target},MODEL_DIR/"aki_gru.pt")
    report={"model":"GRU","dataset_mode":"real_or_user_supplied_csv_unverified","target":args.target,"split":{"train_patients":len(train_ids),"validation_patients":len(val_ids),"test_patients":len(test_ids),"train_examples":len(ytr),"validation_examples":len(yv),"test_examples":len(yte)},"threshold_source":"validation F1","validation":metrics(yv,pv),"test":metrics(yte,pt),"limitations":["GRU is an experimental benchmark and is not wired into the live API by default.","Verify target derivation and temporal leakage before interpreting metrics.","Research-only; not for clinical decisions."]}
    (REPORT_DIR/"gru_evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
