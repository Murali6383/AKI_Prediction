import numpy as np,pandas as pd
from ..ml.features import make_features
def explain_prediction(bundle,payload):
 try:
  import shap
  x=make_features(pd.DataFrame([payload])); pipe=bundle["pipeline"]; model=pipe.named_steps["model"]; prep=pipe.named_steps["preprocessor"]; transformed=prep.transform(x); names=prep.get_feature_names_out().tolist()
  if hasattr(model,"feature_importances_"):
   exp=shap.TreeExplainer(model); vals=exp.shap_values(transformed); vals=vals[-1] if isinstance(vals,list) else vals
   vals=np.asarray(vals); vals=vals[0,:, -1] if vals.ndim==3 else vals[0] if vals.ndim==2 else vals
   base=exp.expected_value; base=float(np.asarray(base).reshape(-1)[-1])
  else:
   exp=shap.Explainer(model,transformed); out=exp(transformed); vals=np.asarray(out.values); vals=vals[0] if vals.ndim>1 else vals; base=float(np.asarray(out.base_values).reshape(-1)[0])
  pairs=sorted(zip(names,map(float,vals)),key=lambda x:abs(x[1]),reverse=True)[:12]
  return {"method":"SHAP","global_importance":[],"individual":[{"feature":n,"shap_value":round(v,6)} for n,v in pairs],"base_value":base,"warning":"Associations used by the model, not causal evidence."}
 except Exception as e:return {"method":"unavailable","global_importance":[],"individual":[],"warning":"SHAP could not be computed for this input/model.","detail":str(e)[:240]}
