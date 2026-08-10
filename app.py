from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import MODEL1_DIR, MODEL2_DIR, MODEL3_DIR
from inference.power_prediction import PowerPredictor, performance_deviation
from utils.alerts import AssetSignals, alert_level, calculate_risk, health_score, recommendation

st.set_page_config(page_title="SolarTwin AI", page_icon="☀️", layout="wide")

st.markdown("# ☀️ SolarTwin AI")
st.caption("AI-driven solar PV fault intelligence, performance monitoring, Digital Twin and predictive maintenance")


def model_available(directory: Path, required: list[str]) -> bool:
    return all((directory / name).exists() for name in required)


m1_ready = model_available(MODEL1_DIR, ["v1_convnext_pvmd.pth", "class_names.json"])
m2_ready = model_available(MODEL2_DIR, ["v1_xgboost_degradation.pkl", "feature_columns.json"])
m3_ready = model_available(MODEL3_DIR, ["v1_xgboost_power.pkl", "feature_columns.json"])

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Open", ["Dashboard", "Fault Detection", "Electrical Analysis", "Power Prediction", "Alert Center", "Digital Twin", "Model Testing Lab", "Reports"])
    st.divider()
    st.write("**Model status**")
    st.write(f"Model 1 — {'🟢 Ready' if m1_ready else '⚪ Train first'}")
    st.write(f"Model 2 — {'🟢 Ready' if m2_ready else '⚪ Train first'}")
    st.write(f"Model 3 — {'🟢 Ready' if m3_ready else '⚪ Train first'}")

if page == "Dashboard":
    st.subheader("Asset Intelligence Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assets", "1,250", "Prototype")
    c2.metric("Healthy", "1,080", "Prototype")
    c3.metric("High Risk", "52", "Prototype")
    c4.metric("Critical", "23", "Prototype")
    st.info("Connect your asset registry and inference outputs to replace prototype dashboard numbers.")
    st.markdown("### System architecture")
    st.code("Model 1: Image → Fault\nModel 2: Electrical/Thermal → Degradation\nModel 3: Environment/Operations → Expected Power\n                 ↓\n        Alert / Risk Engine\n                 ↓\n           Digital Twin", language="text")

elif page == "Fault Detection":
    st.subheader("🔍 Visual PV Fault Detection")
    uploaded = st.file_uploader("Upload a PV module image", type=["png", "jpg", "jpeg"])
    if not m1_ready:
        st.warning("Model 1 is not trained yet. Run training/train_fault_detection.py and place the saved artifacts in models/fault_detection/.")
    if uploaded:
        from PIL import Image
        image = Image.open(uploaded)
        st.image(image, caption="Input PV image", width=500)
        if m1_ready and st.button("Run Fault Detection"):
            from inference.fault_detection import FaultDetector
            detector = FaultDetector(MODEL1_DIR / "v1_convnext_pvmd.pth", MODEL1_DIR / "class_names.json")
            result = detector.predict(image)
            st.success(f"Prediction: {result['class']}")
            st.metric("Confidence", f"{result['confidence']:.2f}%")
            probs = pd.DataFrame({"Class": result["probabilities"].keys(), "Probability": result["probabilities"].values()})
            st.plotly_chart(px.bar(probs, x="Class", y="Probability", range_y=[0, 100]), use_container_width=True)

elif page == "Electrical Analysis":
    st.subheader("⚡ Electrical Performance Analysis")
    uploaded = st.file_uploader("Upload a prepared feature CSV", type=["csv"])
    if not m2_ready:
        st.warning("Model 2 is not trained yet. The exact degradation target and feature extraction must first be confirmed from the PV Mismatch dataset.")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)
        if m2_ready and st.button("Run Electrical Analysis"):
            from inference.electrical_degradation import DegradationPredictor
            predictor = DegradationPredictor(MODEL2_DIR / "v1_xgboost_degradation.pkl", MODEL2_DIR / "feature_columns.json")
            pred = predictor.predict(df)
            st.metric("Predicted degradation / model output", f"{pred.mean():.2f}")
            st.dataframe(pd.DataFrame({"Prediction": pred}), use_container_width=True)

elif page == "Power Prediction":
    st.subheader("☀️ Solar Power Prediction")
    uploaded = st.file_uploader("Upload solar operational CSV", type=["csv"])
    if not m3_ready:
        st.warning("Model 3 is not trained yet. Run training/train_power_prediction.py after inspecting Dataset 3 and selecting the valid target.")
    if uploaded and m3_ready:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)
        if st.button("Predict Expected Power"):
            try:
                predictor = PowerPredictor(MODEL3_DIR / "v1_xgboost_power.pkl", MODEL3_DIR / "feature_columns.json")
                expected = predictor.predict(df)
                st.metric("Mean expected power", f"{expected.mean():.3f}")
                actual_col = st.selectbox("Actual power column (optional)", ["None"] + list(df.columns))
                if actual_col != "None":
                    deviation = performance_deviation(expected, df[actual_col])
                    result = pd.DataFrame({"Expected Power": expected, "Actual Power": df[actual_col], "Deviation %": deviation})
                    st.dataframe(result.head(100), use_container_width=True)
                    fig = px.line(result, y=["Expected Power", "Actual Power"], title="Actual vs Expected Power")
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric("Mean absolute deviation", f"{deviation.abs().mean():.2f}%")
            except Exception as exc:
                st.error(str(exc))

elif page == "Alert Center":
    st.subheader("🚨 Alert Center")
    st.info("The alert engine combines model outputs at the application layer. It does not train a fourth model and does not control physical equipment.")
    col1, col2, col3 = st.columns(3)
    fault = col1.slider("Fault severity", 0.0, 100.0, 0.0)
    confidence = col2.slider("Fault confidence", 0.0, 100.0, 0.0)
    degradation = col3.number_input("Degradation %", 0.0, 100.0, 0.0)
    deviation = st.number_input("Power deviation %", -100.0, 100.0, 0.0)
    signals = AssetSignals(confidence, fault, degradation, deviation)
    risk = calculate_risk(signals)
    health = health_score(risk)
    level = alert_level(risk)
    st.metric("Risk Score", f"{risk}/100")
    st.metric("Health Score", f"{health}/100")
    st.subheader(f"Status: {level}")
    st.write(recommendation(level))

elif page == "Digital Twin":
    st.subheader("🏭 Digital Twin")
    asset_id = st.text_input("Asset ID", "PV-1024")
    c1, c2, c3 = st.columns(3)
    c1.metric("Health Score", "—")
    c2.metric("Risk", "—")
    c3.metric("Status", "Awaiting inference")
    st.info(f"Digital Twin profile for {asset_id}. Connect the asset registry and saved model outputs to populate live state, history and maintenance information.")

elif page == "Model Testing Lab":
    st.subheader("🧪 Model Testing Lab")
    st.write("All external testing uses saved models without retraining. Ground truth is required for quantitative validation.")
    st.markdown("**Model 1:** upload an external image-folder dataset and run `testing/test_fault_detection.py`.\n\n**Model 2:** upload a feature-compatible CSV and run `testing/test_degradation.py`.\n\n**Model 3:** upload a feature-compatible CSV and run `testing/test_power_prediction.py`.")

elif page == "Reports":
    st.subheader("📊 Reports")
    st.info("Report generation will consume saved inference results. The first release provides the data and model interfaces needed to add PDF/CSV report generation without changing the model layer.")

st.divider()
st.caption("SolarTwin AI — research prototype. Risk thresholds are configurable prototype rules, not universal industrial standards.")
