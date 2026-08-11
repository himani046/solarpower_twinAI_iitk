"""Train SolarTwin AI Model 2 from the PV-Mismatch thermal measurements.

The dataset contains thermal CSV files whose bytes are not always UTF-8 (the
previous version skipped these files). This version decodes UTF-16/UTF-8 variants,
extracts numeric thermal matrices, creates leakage-safe tile-level features, and
uses the filename labels Clean/Dirt/Shadow.

Two tasks are evaluated:
1) Healthy vs Defective: Clean = healthy, Dirt/Shadow = defective.
2) Fault type: Clean/Dirt/Shadow.

Splits are ALWAYS made by source thermal file, never by individual tiles, so tiles
from the same panel cannot leak between train and test. Accuracy, precision,
recall, F1, balanced accuracy, ROC-AUC/PR-AUC (binary), classification reports,
and confusion matrices are saved. A secondary Isolation Forest risk is also saved.
"""
from __future__ import annotations
import json, random, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED=42
random.seed(SEED); np.random.seed(SEED)
REPO=Path(__file__).resolve().parents[1]
RESULTS=REPO/"results"/"model2"; MODEL_DIR=REPO/"models"/"electrical_degradation"
DATASET="himani04012007/pv-mismatch"
LABELS={"clean":0,"dirt":1,"shadow":2}

def write_json(p:Path,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,default=str),encoding="utf-8")

def decode_csv(path:Path):
    """Read text CSVs including UTF-16/BOM files; never silently skip thermal CSVs."""
    encodings=["utf-8-sig","utf-16","utf-16-le","utf-16-be","cp1252","latin1"]
    last=None
    for enc in encodings:
        try:
            df=pd.read_csv(path,header=None,encoding=enc)
            # Keep files that contain a meaningful amount of numeric content.
            num=df.apply(pd.to_numeric,errors="coerce")
            if num.notna().sum().sum() >= max(4, int(num.size*0.25)):
                return num.dropna(axis=0,how="all").dropna(axis=1,how="all")
        except Exception as e: last=e
    raise ValueError(f"Could not decode numeric thermal CSV: {last}")

def thermal_features(mat, tile=16):
    """Create compact features from non-overlapping thermal tiles."""
    a=np.asarray(mat,dtype=float); a=a[np.isfinite(a)] if a.ndim==1 else a
    if a.ndim!=2 or min(a.shape)<2: return []
    # Remove empty/non-numeric rows/columns.
    a=np.nan_to_num(a,nan=np.nanmedian(a) if np.isfinite(a).any() else 0.0,posinf=0.0,neginf=0.0)
    h,w=a.shape; out=[]
    # Adaptive tile size prevents enormous sample counts while preserving spatial patterns.
    ts=max(8,min(tile,h,w))
    for r in range(0,h-ts+1,ts):
        for c in range(0,w-ts+1,ts):
            z=a[r:r+ts,c:c+ts].astype(float)
            gx=np.diff(z,axis=1); gy=np.diff(z,axis=0)
            q=np.percentile(z,[5,25,50,75,95])
            mean=z.mean(); std=z.std()+1e-8
            feats=[mean,std,z.min(),z.max(),*q,
                   np.mean(np.abs(gx)),np.mean(np.abs(gy)),
                   np.mean(gx**2),np.mean(gy**2),
                   np.mean(z>mean+2*std),np.mean(z>q[4]),
                   float(r/h),float(c/w)]
            out.append(feats)
    return out

def load_records(root):
    records=[]; skipped=[]
    for p in sorted(root.rglob("*.csv")):
        name=p.stem.lower()
        label=None
        for key in LABELS:
            if re.search(rf"(^|[_ -]){key}($|[_ -])",name): label=key; break
        if label is None:
            continue
        try:
            mat=decode_csv(p)
            feats=thermal_features(mat)
            if not feats: raise ValueError(f"no usable thermal matrix, shape={mat.shape}")
            for i,f in enumerate(feats):
                records.append({"features":f,"label":label,"source":str(p),"tile":i})
        except Exception as e:
            skipped.append({"file":str(p),"label":label,"error":str(e)})
    return records,skipped

def metrics_binary(y,p,prob):
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "precision":float(precision_score(y,p,zero_division=0)),"recall":float(recall_score(y,p,zero_division=0)),
            "f1":float(f1_score(y,p,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else None,
            "pr_auc":float(average_precision_score(y,prob)) if len(np.unique(y))==2 else None,
            "confusion_matrix":confusion_matrix(y,p).tolist(),
            "classification_report":classification_report(y,p,target_names=["Healthy","Defective"],output_dict=True,zero_division=0)}

def main():
    print("SolarTwin AI — Model 2 (thermal supervised + anomaly risk)")
    import kagglehub
    root=Path(kagglehub.dataset_download(DATASET)); print("Dataset:",root)
    records,skipped=load_records(root)
    if not records: raise RuntimeError("No labelled thermal CSVs could be decoded.")
    df=pd.DataFrame(records); X=np.asarray(df.features.tolist(),dtype=float); y_type=df.label.map(LABELS).to_numpy(); groups=df.source.to_numpy(); y_bin=(y_type>0).astype(int)
    unique_groups=np.unique(groups)
    # 70/30 file-level split. If enough files exist, validation is a further 15% of all files.
    gss=GroupShuffleSplit(n_splits=1,test_size=.30,random_state=SEED); train_idx,test_idx=next(gss.split(X,y_bin,groups))
    train_groups=groups[train_idx]; gss2=GroupShuffleSplit(n_splits=1,test_size=.2143,random_state=SEED); tr2,va2=next(gss2.split(X[train_idx],y_bin[train_idx],train_groups))
    tr_idx=train_idx[tr2]; va_idx=train_idx[va2]
    scaler=StandardScaler(); Xtr=scaler.fit_transform(X[tr_idx]); Xte=scaler.transform(X[test_idx])
    # Binary healthy/defective classifier.
    clf=RandomForestClassifier(n_estimators=500,class_weight="balanced",random_state=SEED,n_jobs=-1,max_features="sqrt",min_samples_leaf=2)
    clf.fit(Xtr,y_bin[tr_idx]); prob=clf.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
    binary=metrics_binary(y_bin[test_idx],pred,prob)
    # Fault type classifier: clean/dirt/shadow.
    type_clf=RandomForestClassifier(n_estimators=500,class_weight="balanced",random_state=SEED,n_jobs=-1,max_features="sqrt",min_samples_leaf=2)
    type_clf.fit(Xtr,y_type[tr_idx]); type_pred=type_clf.predict(Xte)
    type_metrics={"accuracy":float(accuracy_score(y_type[test_idx],type_pred)),"balanced_accuracy":float(balanced_accuracy_score(y_type[test_idx],type_pred)),"precision_macro":float(precision_score(y_type[test_idx],type_pred,average="macro",zero_division=0)),"recall_macro":float(recall_score(y_type[test_idx],type_pred,average="macro",zero_division=0)),"f1_macro":float(f1_score(y_type[test_idx],type_pred,average="macro",zero_division=0)),"confusion_matrix":confusion_matrix(y_type[test_idx],type_pred).tolist(),"classification_report":classification_report(y_type[test_idx],type_pred,target_names=["Clean","Dirt","Shadow"],output_dict=True,zero_division=0)}
    # Secondary unsupervised risk model, fit only on training tiles.
    iso=IsolationForest(n_estimators=300,contamination=.05,random_state=SEED,n_jobs=-1).fit(Xtr); raw=-iso.score_samples(Xte)
    lo,hi=np.percentile(-iso.score_samples(Xtr),1),np.percentile(-iso.score_samples(Xtr),99); risk=np.clip((raw-lo)/max(hi-lo,1e-9)*100,0,100)
    metrics={"task":"thermal_pv_anomaly_detection","dataset":DATASET,"decoded_labelled_files":int(len(unique_groups)),"usable_tiles":int(len(df)),"skipped_files":skipped,"split":"group-aware 70/15/15 by source thermal file","healthy_defective":binary,"fault_type":type_metrics,"risk":{"mean":float(risk.mean()),"median":float(np.median(risk)),"p95":float(np.percentile(risk,95)),"flag_rate_at_50":float((risk>=50).mean())}}
    MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    joblib.dump(clf,MODEL_DIR/"v3_random_forest_healthy_defective.pkl"); joblib.dump(type_clf,MODEL_DIR/"v3_random_forest_fault_type.pkl"); joblib.dump(iso,MODEL_DIR/"v3_isolation_forest.pkl"); joblib.dump(scaler,MODEL_DIR/"thermal_scaler.pkl")
    write_json(MODEL_DIR/"metadata.json",{"feature_count":X.shape[1],"labels":LABELS,"threshold":.5,"split":"group-aware by source file"})
    write_json(RESULTS/"metrics.json",metrics); write_json(RESULTS/"skipped_files.json",{"skipped":skipped})
    pd.DataFrame({"source":groups[test_idx],"true_label":[["Clean","Dirt","Shadow"][i] for i in y_type[test_idx]],"healthy_defective_probability":prob,"predicted_healthy_defective":pred,"predicted_fault_type":[["Clean","Dirt","Shadow"][i] for i in type_pred],"risk":risk}).to_csv(RESULTS/"test_predictions.csv",index=False)
    print("\nMODEL 2 COMPLETE")
    print("Decoded labelled files:",len(unique_groups)," | Tiles:",len(df)," | Skipped:",len(skipped))
    print("\nHEALTHY vs DEFECTIVE")
    for k in ["accuracy","balanced_accuracy","precision","recall","f1","roc_auc","pr_auc"]: print(f"{k:22s}: {binary[k]:.4f}" if binary[k] is not None else f"{k:22s}: N/A")
    print("Confusion Matrix:\n",np.asarray(binary["confusion_matrix"]))
    print("\nFAULT TYPE (Clean/Dirt/Shadow)")
    for k in ["accuracy","balanced_accuracy","precision_macro","recall_macro","f1_macro"]: print(f"{k:22s}: {type_metrics[k]:.4f}")
    print("Confusion Matrix:\n",np.asarray(type_metrics["confusion_matrix"]))
    print("\nRisk: mean={:.2f}, median={:.2f}, P95={:.2f}, flag@50={:.2%}".format(risk.mean(),np.median(risk),np.percentile(risk,95),(risk>=50).mean()))
if __name__=="__main__": main()
