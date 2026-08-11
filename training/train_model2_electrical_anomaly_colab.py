"""Train SolarTwin AI Model 2: unsupervised electrical/thermal anomaly detector."""
from __future__ import annotations
import json, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED=42
random.seed(SEED); np.random.seed(SEED)
REPO=Path(__file__).resolve().parents[1]
RESULTS=REPO/"results"/"model2"
MODEL_DIR=REPO/"models"/"electrical_degradation"
DATASET="himani04012007/pv-mismatch"

def write_json(p:Path,x:dict):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,default=str),encoding="utf-8")

def read_csv(path):
    df=pd.read_csv(path)
    numeric_like=0
    for c in df.columns:
        try: float(str(c)); numeric_like+=1
        except: pass
    if len(df.columns)>1 and numeric_like/len(df.columns)>0.8:
        df=pd.read_csv(path,header=None); df.columns=[f"feature_{i}" for i in range(df.shape[1])]
    return df

def main():
    print("SolarTwin AI — Model 2")
    import kagglehub
    root=Path(kagglehub.dataset_download(DATASET)); print("Dataset:",root)
    files=sorted(root.rglob("*.csv"))
    if not files: raise FileNotFoundError("No CSV files found")
    frames=[]
    for f in files:
        try:
            x=read_csv(f)
            if len(x): frames.append(x)
        except Exception as e: print("Skipped",f,e)
    df=pd.concat(frames,ignore_index=True,sort=False)
    numeric=df.select_dtypes(include="number").replace([np.inf,-np.inf],np.nan)
    numeric=numeric.dropna(axis=1,how="all")
    numeric=numeric.loc[:,numeric.nunique(dropna=True)>1]
    numeric=numeric.fillna(numeric.median(numeric_only=True)).dropna(axis=1,how="any")
    train,test=train_test_split(numeric,test_size=.20,random_state=SEED,shuffle=True)
    scaler=StandardScaler(); Xtr=scaler.fit_transform(train); Xte=scaler.transform(test)
    model=IsolationForest(n_estimators=300,contamination=.05,random_state=SEED,n_jobs=-1).fit(Xtr)
    tr_raw=-model.score_samples(Xtr); te_raw=-model.score_samples(Xte)
    lo,hi=np.percentile(tr_raw,1),np.percentile(tr_raw,99); d=max(hi-lo,1e-9)
    tr_risk=np.clip((tr_raw-lo)/d*100,0,100); te_risk=np.clip((te_raw-lo)/d*100,0,100)
    metrics={"task":"unsupervised electrical_thermal_anomaly_detection","dataset":DATASET,"rows":len(numeric),"features":list(numeric.columns),"train_rows":len(train),"test_rows":len(test),"train_flag_rate":float((tr_risk>=50).mean()),"test_flag_rate":float((te_risk>=50).mean()),"train_mean_risk":float(tr_risk.mean()),"test_mean_risk":float(te_risk.mean()),"test_median_risk":float(np.median(te_risk)),"test_p95_risk":float(np.percentile(te_risk,95)),"ground_truth_available":False,"accuracy":None,"precision":None,"recall":None,"f1":None,"roc_auc":None,"pr_auc":None,"confusion_matrix":None,"note":"No validated target exists; classification metrics are intentionally null."}
    MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,MODEL_DIR/"v2_isolation_forest.pkl"); joblib.dump(scaler,MODEL_DIR/"scaler.pkl")
    write_json(MODEL_DIR/"feature_columns.json",{"features":list(numeric.columns)})
    write_json(MODEL_DIR/"preprocessing.json",{"imputation":"training median","scaler":"StandardScaler","p1":float(lo),"p99":float(hi),"risk_threshold":50})
    write_json(MODEL_DIR/"model_metadata.json",metrics); write_json(RESULTS/"metrics.json",metrics)
    print(json.dumps(metrics,indent=2)); print("Model 2 complete.")
if __name__=="__main__": main()
