"""SolarTwin AI Model 3: expected AC-power prediction and power-deviation analysis."""
from __future__ import annotations
import json, math, random, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score, max_error
from xgboost import XGBRegressor

SEED=42
random.seed(SEED); np.random.seed(SEED)
REPO=Path(__file__).resolve().parents[1]
RESULTS=REPO/"results"/"model3"; MODEL_DIR=REPO/"models"/"power_prediction"
DATASET="anikannal/solar-power-generation-data"

def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")

def parse_dt(series):
    s=series.astype(str).str.strip()
    out=pd.to_datetime(s,format="%d-%m-%Y %H:%M",errors="coerce")
    miss=out.isna(); out.loc[miss]=pd.to_datetime(s.loc[miss],format="%Y-%m-%d %H:%M:%S",errors="coerce")
    miss=out.isna(); out.loc[miss]=pd.to_datetime(s.loc[miss],format="%d/%m/%Y %H:%M",errors="coerce")
    return out

def find_files(root):
    generation=[]; weather=[]
    for p in sorted(root.rglob("*.csv")):
        try: cols={str(c).strip().upper() for c in pd.read_csv(p,nrows=2).columns}
        except Exception: continue
        if "AC_POWER" in cols and "DATE_TIME" in cols: generation.append(p)
        if "IRRADIATION" in cols and "DATE_TIME" in cols: weather.append(p)
    if not generation: raise FileNotFoundError("No generation CSV found.")
    gp=generation[0]; m=re.search(r"PLANT[_ -]?(\d+)",gp.name,re.I); plant=m.group(1) if m else None
    wp=None
    if plant:
        matches=[p for p in weather if re.search(rf"PLANT[_ -]?{plant}",p.name,re.I)]
        if matches: wp=matches[0]
    if wp is None and weather: wp=weather[0]
    return gp,wp

def metrics(actual,pred):
    actual=np.asarray(actual); pred=np.asarray(pred); mse=mean_squared_error(actual,pred); nz=np.abs(actual)>1e-6
    return {"MAE":float(mean_absolute_error(actual,pred)),"MSE":float(mse),"RMSE":float(np.sqrt(mse)),"R2":float(r2_score(actual,pred)),"MAPE_percent_nonzero_actual":float(np.mean(np.abs((actual[nz]-pred[nz])/actual[nz]))*100) if nz.any() else None,"MedianAbsoluteError":float(median_absolute_error(actual,pred)),"ExplainedVariance":float(explained_variance_score(actual,pred)),"MaxError":float(max_error(actual,pred))}

def main():
    print("SolarTwin AI — Model 3 (Expected Power Prediction)")
    import kagglehub
    root=Path(kagglehub.dataset_download(DATASET)); gp,wp=find_files(root)
    print("Dataset:",root); print("Generation:",gp); print("Weather:",wp)
    g=pd.read_csv(gp); g.columns=[str(c).strip().upper() for c in g.columns]; g["DATE_TIME"]=parse_dt(g["DATE_TIME"]); g["AC_POWER"]=pd.to_numeric(g["AC_POWER"],errors="coerce"); g=g.dropna(subset=["DATE_TIME","AC_POWER"])
    if wp is not None:
        w=pd.read_csv(wp); w.columns=[str(c).strip().upper() for c in w.columns]; w["DATE_TIME"]=parse_dt(w["DATE_TIME"]); w=w.dropna(subset=["DATE_TIME"])
        keys=["DATE_TIME"]+(["PLANT_ID"] if "PLANT_ID" in g.columns and "PLANT_ID" in w.columns else [])
        cols=[c for c in ["DATE_TIME","PLANT_ID","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"] if c in w.columns]
        w=w[cols].drop_duplicates(keys); df=g.merge(w,on=keys,how="left")
    else: df=g.copy()
    df=df.sort_values("DATE_TIME").reset_index(drop=True)
    df["hour"]=df.DATE_TIME.dt.hour; df["minute"]=df.DATE_TIME.dt.minute; df["day_of_week"]=df.DATE_TIME.dt.dayofweek; df["day_of_year"]=df.DATE_TIME.dt.dayofyear; df["month"]=df.DATE_TIME.dt.month
    t=df.hour+df.minute/60; df["hour_sin"]=np.sin(2*math.pi*t/24); df["hour_cos"]=np.cos(2*math.pi*t/24); df["day_sin"]=np.sin(2*math.pi*df.day_of_year/365.25); df["day_cos"]=np.cos(2*math.pi*df.day_of_year/365.25)
    features=[c for c in ["AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION","hour","minute","day_of_week","day_of_year","month","hour_sin","hour_cos","day_sin","day_cos"] if c in df.columns]
    x=df[features+["AC_POWER","DATE_TIME"]].copy()
    for c in features: x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["AC_POWER"]).reset_index(drop=True)
    split=int(len(x)*.80); train=x.iloc[:split].copy(); test=x.iloc[split:].copy()
    med=train[features].median(); train[features]=train[features].fillna(med); test[features]=test[features].fillna(med)
    model=XGBRegressor(n_estimators=600,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective="reg:squarederror",random_state=SEED,n_jobs=-1); model.fit(train[features],train["AC_POWER"])
    actual=test["AC_POWER"].to_numpy(); pred=model.predict(test[features]); overall=metrics(actual,pred)
    baseline=np.full_like(actual,train["AC_POWER"].mean(),dtype=float); baseline_metrics=metrics(actual,baseline)
    daylight=test["IRRADIATION"].to_numpy()>20 if "IRRADIATION" in test.columns else actual>1
    daylight_metrics=metrics(actual[daylight],pred[daylight]) if daylight.any() else {}
    out=pd.DataFrame({"date_time":test.DATE_TIME.astype(str),"actual_power":actual,"expected_power":pred,"residual":actual-pred,"absolute_error":np.abs(actual-pred),"daylight":daylight})
    out["deviation_percent"]=(out.actual_power-out.expected_power)/out.expected_power.abs().replace(0,np.nan)*100; out["deviation_percent"]=out.deviation_percent.replace([np.inf,-np.inf],np.nan).fillna(0)
    payload={"task":"expected_ac_power_prediction","dataset":DATASET,"generation_file":str(gp),"weather_file":str(wp) if wp else None,"features":features,"dc_power_excluded":True,"train_rows":len(train),"test_rows":len(test),"daylight_test_rows":int(daylight.sum()),"split":"chronological 80:20","model":overall,"daylight_only":daylight_metrics,"naive_train_mean_baseline":baseline_metrics,"note":"Weather file is matched to the generation plant when possible. Median imputation is fit on training data only. DC_POWER is excluded to avoid direct target leakage. Daylight uses IRRADIATION > 20."}
    MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,MODEL_DIR/"v3_xgboost_power_no_dc.pkl"); write_json(MODEL_DIR/"feature_columns.json",{"features":features}); write_json(MODEL_DIR/"preprocessing.json",{"numeric_medians":med.to_dict(),"dc_power_excluded":True,"daylight_irradiation_threshold":20}); write_json(MODEL_DIR/"model_metadata.json",payload); write_json(RESULTS/"metrics.json",payload); out.to_csv(RESULTS/"test_predictions.csv",index=False)
    print("\n================ MODEL 3 COMPLETE ================")
    print("ALL TEST SAMPLES"); [print(f"{k:30s}: {v}") for k,v in overall.items()]
    print("\nDAYLIGHT ONLY (IRRADIATION > 20)"); [print(f"{k:30s}: {v}") for k,v in daylight_metrics.items()]
    print("\nNAIVE BASELINE"); [print(f"{k:30s}: {v}") for k,v in baseline_metrics.items()]
    print("\nModel saved:",MODEL_DIR/"v3_xgboost_power_no_dc.pkl")
if __name__=="__main__": main()
