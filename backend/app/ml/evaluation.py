import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,precision_score,recall_score,f1_score,brier_score_loss,roc_curve,precision_recall_curve
from sklearn.calibration import calibration_curve
def evaluate_probabilities(y_true,probs,threshold):
    y=np.asarray(y_true).astype(int); p=np.asarray(probs).astype(float); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    def safe(fn):
        try:return float(fn(y,p))
        except Exception:return None
    fpr,tpr,_=roc_curve(y,p) if len(np.unique(y))>1 else ([],[],[])
    pr,rc,_=precision_recall_curve(y,p)
    calx,caly=calibration_curve(y,p,n_bins=min(10,max(2,len(y)//10)),strategy="quantile") if len(y)>=20 else ([],[])
    return {"auroc":safe(roc_auc_score),"auprc":safe(average_precision_score),"sensitivity":float(tp/(tp+fn)) if tp+fn else None,"specificity":float(tn/(tn+fp)) if tn+fp else None,"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"brier_score":float(brier_score_loss(y,p)),"threshold":float(threshold),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]],"roc_curve":[{"fpr":float(a),"tpr":float(b)} for a,b in zip(fpr,tpr)],"pr_curve":[{"recall":float(a),"precision":float(b)} for a,b in zip(rc,pr)],"calibration":[{"mean_predicted":float(a),"fraction_positive":float(b)} for a,b in zip(calx,caly)]}
