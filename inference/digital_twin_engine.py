from __future__ import annotations
from pathlib import Path
from model1_visual_fault import Model1VisualFault
from model3_power_prediction import Model3PowerPrediction

ROOT=Path(__file__).resolve().parents[1]

class DigitalTwinEngine:
    """Transparent integration layer; Model 2 remains optional/experimental."""
    def __init__(self, use_model1=True, use_model3=True):
        self.model1=Model1VisualFault() if use_model1 and (ROOT/"models/fault_detection/v2_convnext_pvmd_multilabel.pth").exists() else None
        self.model3=Model3PowerPrediction() if use_model3 and (ROOT/"models/power_prediction/v3_xgboost_power_no_dc.pkl").exists() else None
    def analyze(self, panel_id, image_path=None, power=None, timestamp=None, ambient_temperature=None, module_temperature=None, irradiation=None, model2_risk=None):
        visual=None
        if self.model1 and image_path: visual=self.model1.predict(image_path)
        power_state=None
        if self.model3 and timestamp is not None and ambient_temperature is not None and module_temperature is not None and irradiation is not None:
            power_state=self.model3.predict(timestamp,ambient_temperature,module_temperature,irradiation)
            if power is not None:
                power_state["actual_ac_power"]=float(power)
                power_state["deviation_percent"]=self.model3.deviation(power,power_state["expected_ac_power"])
        faults=len(visual["anomalies"]) if visual else 0
        deviation=abs(power_state.get("deviation_percent",0)) if power_state else 0
        risk=float(model2_risk) if model2_risk is not None else 0.0
        if faults and deviation>=20: alert="RED"
        elif faults or risk>=70 or deviation>=20: alert="ORANGE"
        elif risk>=50 or deviation>=10: alert="YELLOW"
        else: alert="GREEN"
        return {"panel_id":panel_id,"visual_fault":visual,"model2_risk":risk if model2_risk is not None else None,"power":power_state,"alert_level":alert,"model2_note":"experimental secondary signal; not a validated fault probability"}

if __name__=="__main__":
    print("SolarTwin AI Digital Twin Engine")
    print("Import DigitalTwinEngine and call analyze(...) with panel/image/power/weather inputs.")
