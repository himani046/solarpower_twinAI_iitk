from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"models/power_prediction/v3_xgboost_power_no_dc.pkl"
FEATURE_PATH=ROOT/"models/power_prediction/feature_columns.json"
PREP_PATH=ROOT/"models/power_prediction/preprocessing.json"

class Model3PowerPrediction:
    def __init__(self):
        self.model=joblib.load(MODEL_PATH)
        self.features=json.loads(FEATURE_PATH.read_text())["features"]
        prep=json.loads(PREP_PATH.read_text())
        self.medians=prep.get("numeric_medians",{})
    def predict(self, timestamp, ambient_temperature, module_temperature, irradiation):
        ts=pd.Timestamp(timestamp)
        hour=ts.hour; minute=ts.minute; doy=ts.dayofyear; dow=ts.dayofweek; month=ts.month
        t=hour+minute/60
        row={"AMBIENT_TEMPERATURE":ambient_temperature,"MODULE_TEMPERATURE":module_temperature,"IRRADIATION":irradiation,"hour":hour,"minute":minute,"day_of_week":dow,"day_of_year":doy,"month":month,"hour_sin":np.sin(2*np.pi*t/24),"hour_cos":np.cos(2*np.pi*t/24),"day_sin":np.sin(2*np.pi*doy/365.25),"day_cos":np.cos(2*np.pi*doy/365.25)}
        x=pd.DataFrame([{c:row.get(c,self.medians.get(c,0.0)) for c in self.features}])
        expected=float(self.model.predict(x)[0])
        return {"expected_ac_power":expected,"irradiation":float(irradiation),"daylight":bool(float(irradiation)>20)}
    @staticmethod
    def deviation(actual, expected):
        actual=float(actual); expected=float(expected)
        return float((expected-actual)/max(abs(expected),1e-6)*100)
