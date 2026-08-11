"""Train SolarTwin AI Model 3: expected AC power predictor without DC_POWER leakage."""
from __future__ import annotations
import json, math, random
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,median_absolute_error,explained_variance_score,max_error
from xgboost import XGBRegressor

SEED=42; random.seed(SEED); np.random.seed(SEED)
REPO=Path(__file__).resolve().parents[1]; RESULTS=REPO/"results"/"model3"; MODEL_DIR=REPO/"models"/"power_prediction"
DATASET="anikannal/solar-power-generation-data"

def write_json(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,default=str),encoding="utf-8")
def parse_dt(s):
    out=pd.to_datetime(s,format="%d-%m-%Y %H:%M",errors="coerce")
    m=out.isna(); out.loc[m]=pd.to_datetime(s.loc[m],format="%Y-%m-%d %H:%M:%S",errors="coerce")
    m=out.isna(); out.loc[m]=pd.to_datetime(s.loc[m],errors="coerce")
    return out

def find_files(root):
    gen=weather=None
    for p in sorted(root.rglob("*.csv")):
        try: cols={str(c).strip().upper() for c in pd.read_csv(p,nrows=2).columns}
        except: continue
        if "AC_POWER" in cols and "DATE_TIME" in cols: gen=p
        if "IRRADIATION" in cols and "DATE_TIME" in cols: weather=p
    if gen is None: raise FileNotFoundError("Generation CSV not found")
    return gen,weather

def main():
    print("SolarTwin AI — Model 3")
    import kagglehub
    root=Path(kagglehub.dataset_download(DATASET)); print("Dataset:",root)
    gp,wp=find_files(root); g=pd.read_csv(gp); g.columns=[str(c).strip().upper() for c in g.columns]; g["DATE_TIME"]=parse_dt(g["DATE_TIME"]); g=g.dropna(subset=["DATE_TIME","AC_POWER"])
    if wp:
        w=pd.read_csv(wp); w.columns=[str(c).strip().upper() for c in w.columns]; w["DATE_TIME"]=parse_dt(w["DATE_TIME"]); w=w.dropna(subset=["DATE_TIME"])
        keys=["DATE_TIME"]+(["PLANT_ID"] if "PLANT_ID" in g.columns and "PLANT_ID" in w.columns else [])
        cols=[c for c in ["DATE_TIME","PLANT_ID","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"] if c in w.columns]
        w=w[cols].drop_duplicates(keys); df=g.merge(w,on=keys,how="left")
    else: df=g.copy()
    df=df.sort_values("DATE_TIME").reset_index(drop=True)
    df["hour"]=df.DATE_TIME.dt.hour; df["minute"]=df.DATE_TIME.dt.minute; df["day_of_week"]=df.DATE_TIME.dt.dayofweek; df["day_of_year"]=df.DATE_TIME.dt.dayofyear; df["month"]=df.DATE_TIME.dt.month
    t=df.hour+df.minute/60; df["hour_sin"]=np.sin(2*math.pi*t/24); df["hour_cos"]=np.cos(2*math.pi*t/24); df["day_sin"]=np.sin(2*math.pi*df.day_of_year/365.25); df["day_cos"]=np.cos(2*math.pi*df.day_of_year/365.25)
    features=[c for c in ["AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION","hour","minute","day_of_week","day_of_year","month","hour_sin","hour_cos","day_sin","day_cos"] if c in df.columns]
    x=df[features+['AC_POWER']].copy()
    for c in features: x[c]=pd.to_numeric(x[c],errors='coerce')
    x['AC_POWER']=pd.to_numeric(x['AC_POWER'],errors='coerce'); x=x.dropna(subset=['AC_POWER']); med=x[features].median(); x[features]=x[features].fillna(med)
    split=int(len(x)*.8); train=x.iloc[:split]; test=x.iloc[split:]
    model=XGBRegressor(n_estimators=500,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='reg:squarederror',random_state=SEED,n_jobs=-1); model.fit(train[features],train.AC_POWER)
    actual=test.AC_POWER.to_numpy(); pred=model.predict(test[features]); mse=mean_squared_error(actual,pred); nz=np.abs(actual)>1e-6
    metrics={"task":"expected_ac_power_prediction","dataset":DATASET,"features":features,"dc_power_excluded":True,"train_rows":len(train),"test_rows":len(test),"MAE":float(mean_absolute_error(actual,pred)),"MSE":float(mse),"RMSE":float(np.sqrt(mse)),"R2":float(r2_score(actual,pred)),"MAPE_percent_nonzero_actual":float(np.mean(np.abs((actual[nz]-pred[nz])/actual[nz]))*100),"MedianAbsoluteError":float(median_absolute_error(actual,pred)),"ExplainedVariance":float(explained_variance_score(actual,pred)),"MaxError":float(max_error(actual,pred)),"split":"chronological 80:20"}
    MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True); joblib.dump(model,MODEL_DIR/"v2_xgboost_power_no_dc.pkl"); write_json(MODEL_DIR/"feature_columns.json",{"features":features}); write_json(MODEL_DIR/"preprocessing.json",{"numeric_medians":med.to_dict(),"dc_power_excluded":True}); write_json(MODEL_DIR/"model_metadata.json",metrics); write_json(RESULTS/"metrics.json",metrics)
    pd.DataFrame({"actual_power":actual,"expected_power":pred,"residual":actual-pred,"absolute_error":np.abs(actual-pred)}).to_csv(RESULTS/"test_predictions.csv",index=False)
    print("\nMODEL 3 COMPLETE METRICS"); [print(f"{k:30s}: {metrics[k]}") for k in ["MAE","MSE","RMSE","R2","MAPE_percent_nonzero_actual","MedianAbsoluteError","ExplainedVariance","MaxError"]]; print("Model 3 complete.")
if __name__=="__main__": main()
