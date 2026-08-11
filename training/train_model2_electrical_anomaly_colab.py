"""Train SolarTwin AI Model 2 at whole-panel thermal-file level.

The previous version created many tiles from only 18 independent thermal files.
This version treats each thermal CSV as one independent panel sample, extracts
whole-panel thermal statistics, and evaluates with Leave-One-Source-File-Out
cross-validation. It reports honest out-of-file metrics and then fits production
models on all decoded files.

Clean -> Healthy
Dirt/Shadow -> Defective
"""
from __future__ import annotations
import json, random, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED=42; random.seed(SEED); np.random.seed(SEED)
REPO=Path(__file__).resolve().parents[1]
RESULTS=REPO/"results"/"model2"; MODEL_DIR=REPO/"models"/"electrical_degradation"
DATASET="himani04012007/pv-mismatch"; LABELS={"clean":0,"dirt":1,"shadow":2}

def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,default=str),encoding="utf-8")

def decode_csv(path):
    errors=[]
    for enc in ["utf-8-sig","utf-16","utf-16-le","utf-16-be","cp1252","latin1"]:
        try:
            raw=pd.read_csv(path,header=None,encoding=enc); num=raw.apply(pd.to_numeric,errors="coerce")
            if num.notna().sum().sum() >= max(4,int(num.size*.25)):
                num=num.dropna(axis=0,how="all").dropna(axis=1,how="all")
                if num.shape[0]>=2 and num.shape[1]>=2:return num
        except Exception as e: errors.append(f"{enc}: {e}")
    raise ValueError("No usable numeric matrix: "+" | ".join(errors))

def panel_features(mat):
    a=np.asarray(mat,dtype=float)
    if a.ndim!=2 or min(a.shape)<2:return None
    fill=float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
    a=np.nan_to_num(a,nan=fill,posinf=fill,neginf=fill); flat=a.ravel(); mean=flat.mean(); std=flat.std()+1e-8
    q=np.percentile(flat,[1,5,10,25,50,75,90,95,99]); gx=np.diff(a,axis=1); gy=np.diff(a,axis=0)
    return [float(mean),float(std),float(flat.min()),float(flat.max()),*map(float,q),float(np.mean(np.abs(gx))),float(np.mean(np.abs(gy))),float(np.mean(gx**2)),float(np.mean(gy**2)),float(np.mean(flat>mean+std)),float(np.mean(flat>mean+2*std)),float(np.mean(flat>mean+3*std)),float(np.mean(flat>=q[-1])),float(np.std(np.mean(a,axis=0))),float(np.std(np.mean(a,axis=1))),float(np.ptp(np.mean(a,axis=0))),float(np.ptp(np.mean(a,axis=1))),float(a.shape[0]),float(a.shape[1]),float(a.shape[0]/max(a.shape[1],1))]

def label_from_filename(path):
    stem=path.stem.lower()
    for k in LABELS:
        if re.search(rf"(^|[_ -]){k}($|[_ -])",stem): return k
    return None

def load_records(root):
    records=[]; skipped=[]; files=list(root.rglob("*.csv"))
    for p in sorted(files):
        label=label_from_filename(p)
        if label is None: continue
        try:
            f=panel_features(decode_csv(p))
            if f is None: raise ValueError("invalid thermal matrix")
            records.append({"features":f,"label":label,"source":str(p)})
        except Exception as e: skipped.append({"file":str(p),"label":label,"error":str(e)})
    return records,skipped,len(files)

def model(binary=True):
    rf=RandomForestClassifier(n_estimators=500,class_weight="balanced",random_state=SEED,n_jobs=-1,max_features="sqrt",min_samples_leaf=2)
    lr=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("lr",LogisticRegression(class_weight="balanced",max_iter=3000,C=.5,random_state=SEED))])
    return rf,lr

def binary_metrics(y,p,prob):
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"precision":float(precision_score(y,p,zero_division=0)),"recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),"roc_auc":float(roc_auc_score(y,prob)),"pr_auc":float(average_precision_score(y,prob)),"confusion_matrix":confusion_matrix(y,p,labels=[0,1]).tolist(),"classification_report":classification_report(y,p,target_names=["Healthy","Defective"],output_dict=True,zero_division=0)}

def multi_metrics(y,p):
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"precision_macro":float(precision_score(y,p,average="macro",zero_division=0)),"recall_macro":float(recall_score(y,p,average="macro",zero_division=0)),"f1_macro":float(f1_score(y,p,average="macro",zero_division=0)),"confusion_matrix":confusion_matrix(y,p,labels=[0,1,2]).tolist(),"classification_report":classification_report(y,p,target_names=["Clean","Dirt","Shadow"],output_dict=True,zero_division=0)}

def main():
    print("SolarTwin AI — Model 2 (whole-panel thermal)")
    import kagglehub
    root=Path(kagglehub.dataset_download(DATASET)); print("Dataset:",root)
    records,skipped,total=load_records(root)
    if len(records)<6: raise RuntimeError(f"Only {len(records)} labelled files decoded")
    df=pd.DataFrame(records); X=np.asarray(df.features.tolist(),dtype=float); y_type=df.label.map(LABELS).to_numpy(); y_bin=(y_type>0).astype(int); groups=df.source.to_numpy(); n=len(df)
    # Honest OOF evaluation: every source thermal CSV is held out once.
    logo=LeaveOneGroupOut(); bin_prob=np.full(n,np.nan); bin_pred=np.full(n,-1,dtype=int); type_pred=np.full(n,-1,dtype=int)
    for tr,te in logo.split(X,y_bin,groups):
        _,lr=model(True); lr.fit(X[tr],y_bin[tr]); prob=lr.predict_proba(X[te])[:,1]; bin_prob[te]=prob; bin_pred[te]=(prob>=.5).astype(int)
        _,tlr=model(False)
        try: tlr.fit(X[tr],y_type[tr]); type_pred[te]=tlr.predict(X[te])
        except Exception:
            rf,_=model(False); rf.fit(X[tr],y_type[tr]); type_pred[te]=rf.predict(X[te])
    valid=np.isfinite(bin_prob); hd=binary_metrics(y_bin[valid],bin_pred[valid],bin_prob[valid]); ft=multi_metrics(y_type,type_pred)
    # Production classifiers are fit on all independent source files.
    rf_bin,lr_bin=model(True); rf_bin.fit(X,y_bin); lr_bin.fit(X,y_bin); rf_type,lr_type=model(False); rf_type.fit(X,y_type); lr_type.fit(X,y_type)
    iso=IsolationForest(n_estimators=300,contamination=.05,random_state=SEED,n_jobs=-1).fit(X); raw=-iso.score_samples(X); p1,p99=np.percentile(raw,1),np.percentile(raw,99); risk=np.clip((raw-p1)/max(p99-p1,1e-9)*100,0,100)
    metrics={"task":"whole_panel_thermal_anomaly_detection","dataset":DATASET,"total_csv_files_found":total,"decoded_labelled_files":n,"skipped_files":len(skipped),"independent_samples":n,"features_per_panel":X.shape[1],"evaluation":"leave-one-source-file-out OOF","healthy_defective":hd,"fault_type":ft,"risk":{"mean":float(risk.mean()),"median":float(np.median(risk)),"p95":float(np.percentile(risk,95)),"flag_rate_at_50":float((risk>=50).mean())},"warning":"Only the currently decoded labelled source files are available; metrics are exploratory and must not be presented as production validation."}
    MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    joblib.dump(rf_bin,MODEL_DIR/"v4_random_forest_healthy_defective.pkl"); joblib.dump(rf_type,MODEL_DIR/"v4_random_forest_fault_type.pkl"); joblib.dump(lr_bin,MODEL_DIR/"v4_logistic_healthy_defective.pkl"); joblib.dump(lr_type,MODEL_DIR/"v4_logistic_fault_type.pkl"); joblib.dump(iso,MODEL_DIR/"v4_isolation_forest.pkl")
    write_json(MODEL_DIR/"metadata.json",{"feature_count":int(X.shape[1]),"labels":LABELS,"evaluation":"leave-one-source-file-out","production_classifier":"RandomForest","sample_unit":"one thermal CSV = one panel sample"})
    write_json(RESULTS/"metrics.json",metrics); write_json(RESULTS/"skipped_files.json",{"skipped":skipped})
    pd.DataFrame({"source":groups,"true_label":[["Clean","Dirt","Shadow"][i] for i in y_type],"oof_healthy_defective_probability":bin_prob,"oof_predicted_healthy_defective":["Defective" if x==1 else "Healthy" for x in bin_pred],"oof_predicted_fault_type":[["Clean","Dirt","Shadow"][i] if i>=0 else "Unknown" for i in type_pred],"risk":risk}).to_csv(RESULTS/"oof_predictions.csv",index=False)
    print("\nMODEL 2 COMPLETE — WHOLE PANEL")
    print(f"Decoded labelled files: {n} | Independent samples: {n} | Skipped: {len(skipped)}")
    print("\nHEALTHY vs DEFECTIVE (LOSO OOF)")
    for k in ["accuracy","balanced_accuracy","precision","recall","f1","roc_auc","pr_auc"]: print(f"{k:22s}: {hd[k]:.4f}")
    print("Confusion Matrix:\n",np.asarray(hd["confusion_matrix"]))
    print("\nFAULT TYPE (LOSO OOF)")
    for k in ["accuracy","balanced_accuracy","precision_macro","recall_macro","f1_macro"]: print(f"{k:22s}: {ft[k]:.4f}")
    print("Confusion Matrix:\n",np.asarray(ft["confusion_matrix"]))
    print("\nRisk: mean={:.2f}, median={:.2f}, P95={:.2f}, flag@50={:.2%}".format(risk.mean(),np.median(risk),np.percentile(risk,95),(risk>=50).mean()))
if __name__=="__main__": main()
