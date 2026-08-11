from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import MODEL1_DIR, MODEL2_DIR, MODEL3_DIR
from inference.power_prediction import PowerPredictor, performance_deviation
from utils.alerts import AssetSignals, alert_level, calculate_risk, health_score, recommendation

st.set_page_config(
    page_title="SolarTwin AI",
    page_icon="☀️",
    layout="wide",
)

st.markdown("# ☀️ SolarTwin AI")
st.caption(
    "AI-driven solar PV fault intelligence, performance monitoring, Digital Twin and predictive maintenance"
)


def model_available(directory: Path, required: list[str]) -> bool:
    return all((directory / name).exists() for name in required)


m1_ready = model_available(
    MODEL1_DIR,
    [
        "v2_convnext_pvmd_multilabel.pth",
        "class_names.json",
        "preprocessing.json",
    ],
)
m2_ready = model_available(
    MODEL2_DIR,
    ["v1_xgboost_degradation.pkl", "feature_columns.json"],
)
m3_ready = model_available(
    MODEL3_DIR,
    ["v1_xgboost_power.pkl", "feature_columns.json"],
)

if "last_fault_result" not in st.session_state:
    st.session_state.last_fault_result = None


with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Open",
        [
            "Dashboard",
            "Fault Detection",
            "Electrical Analysis",
            "Power Prediction",
            "Alert Center",
            "Digital Twin",
            "Model Testing Lab",
            "Reports",
        ],
    )
    st.divider()
    st.write("**Model status**")
    st.write(f"Model 1 — {'🟢 Ready' if m1_ready else '⚪ Train first'}")
    st.write(f"Model 2 — {'🟢 Ready' if m2_ready else '⚪ Train first'}")
    st.write(f"Model 3 — {'🟢 Ready' if m3_ready else '⚪ Train first'}")


if page == "Dashboard":
    st.subheader("Asset Intelligence Dashboard")

    latest = st.session_state.last_fault_result

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model 1", "READY" if m1_ready else "NOT READY")

    if latest:
        alert_level_name = latest["alert"]["level"]
        c2.metric("Latest Alert", alert_level_name.replace("_", " "))
        c3.metric("Detected Anomalies", len(latest["detected_anomalies"]))
        c4.metric("Severity Score", f'{latest["alert"]["severity_score"]:.0f}/100')
    else:
        c2.metric("Latest Alert", "—")
        c3.metric("Detected Anomalies", "—")
        c4.metric("Severity Score", "—")

    st.markdown("### System architecture")
    st.code(
        "Model 1: Thermal/visual image → Multiple PV anomalies\n"
        "Model 2: Electrical/thermal data → Degradation\n"
        "Model 3: Environment/operations → Expected power\n"
        "                         ↓\n"
        "                  Alert / Risk Engine\n"
        "                         ↓\n"
        "                    Digital Twin",
        language="text",
    )

    if latest:
        st.markdown("### Latest Model 1 assessment")
        if latest["detected_anomalies"]:
            st.error(f'🔴 {latest["alert"]["label"]}')
        else:
            st.warning("🟡 No trained anomaly detected")
            st.caption(latest["note"])

        anomaly_df = pd.DataFrame(
            [
                {
                    "Anomaly": name,
                    "Confidence (%)": score,
                    "Detection threshold (%)": latest["thresholds"][name],
                }
                for name, score in latest["probabilities"].items()
            ]
        )
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

    st.info(
        "The dashboard does not invent production asset counts. Connect an asset registry and the Model 2/3 outputs when those pipelines are ready."
    )


elif page == "Fault Detection":
    st.subheader("🔍 Visual PV Fault Detection")
    st.write(
        "Upload one PV image. Model 1 independently checks for Cracks, Hotspots and Shadings, so one image can contain multiple detected anomalies."
    )

    uploaded = st.file_uploader(
        "Upload a PV module image",
        type=["png", "jpg", "jpeg"],
        key="fault_image",
    )

    if not m1_ready:
        st.warning(
            "Model 1 is not ready. Add the saved checkpoint and preprocessing artifacts to models/fault_detection/."
        )

    if uploaded:
        from PIL import Image

        image = Image.open(uploaded)
        left, right = st.columns([1.2, 1])

        with left:
            st.image(image, caption="Input PV image", use_container_width=True)

        with right:
            st.markdown("### Run assessment")
            if m1_ready and st.button(
                "Run Fault Detection",
                type="primary",
                use_container_width=True,
            ):
                from inference.fault_detection import FaultDetector

                detector = FaultDetector(
                    MODEL1_DIR / "v2_convnext_pvmd_multilabel.pth",
                    MODEL1_DIR / "class_names.json",
                )
                result = detector.predict(image)
                st.session_state.last_fault_result = result

            result = st.session_state.last_fault_result

            if result:
                alert = result["alert"]
                if alert["level"] == "CRITICAL":
                    st.error(f'🚨 {alert["label"]}')
                elif alert["level"] == "HIGH_RISK":
                    st.warning(f'🟠 {alert["label"]}')
                elif alert["level"] == "ATTENTION":
                    st.info(f'🟡 {alert["label"]}')
                else:
                    st.success("🟢 No trained anomaly detected")

                st.metric(
                    "Inspection priority",
                    f'{alert["severity_score"]:.0f}/100',
                )

                st.markdown("### Detected anomalies")
                if result["detected_anomalies"]:
                    for anomaly in result["detected_anomalies"]:
                        st.write(
                            f'**{anomaly["name"]}** — {anomaly["confidence"]:.2f}% confidence '
                            f'(threshold {anomaly["threshold"]:.0f}%)'
                        )
                else:
                    st.write("No trained anomaly crossed its calibrated threshold.")

                st.markdown("### Recommended action")
                st.write(alert["recommendation"])
                st.caption(result["note"])

        if st.session_state.last_fault_result:
            result = st.session_state.last_fault_result
            st.markdown("### Anomaly confidence profile")
            probs = pd.DataFrame(
                {
                    "Anomaly": list(result["probabilities"].keys()),
                    "Confidence": list(result["probabilities"].values()),
                }
            )
            fig = px.bar(
                probs,
                x="Anomaly",
                y="Confidence",
                range_y=[0, 100],
                labels={"Confidence": "Model confidence (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Calibrated detection thresholds")
            threshold_df = pd.DataFrame(
                {
                    "Anomaly": list(result["thresholds"].keys()),
                    "Threshold (%)": list(result["thresholds"].values()),
                }
            )
            st.dataframe(threshold_df, use_container_width=True, hide_index=True)


elif page == "Electrical Analysis":
    st.subheader("⚡ Electrical Performance Analysis")
    uploaded = st.file_uploader("Upload a prepared feature CSV", type=["csv"])
    if not m2_ready:
        st.warning(
            "Model 2 is not trained yet. The exact degradation target and feature extraction must first be confirmed from the PV Mismatch dataset."
        )
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)
        if m2_ready and st.button("Run Electrical Analysis"):
            from inference.electrical_degradation import DegradationPredictor

            predictor = DegradationPredictor(
                MODEL2_DIR / "v1_xgboost_degradation.pkl",
                MODEL2_DIR / "feature_columns.json",
            )
            pred = predictor.predict(df)
            st.metric("Predicted degradation / model output", f"{pred.mean():.2f}")
            st.dataframe(pd.DataFrame({"Prediction": pred}), use_container_width=True)


elif page == "Power Prediction":
    st.subheader("☀️ Solar Power Prediction")
    uploaded = st.file_uploader("Upload solar operational CSV", type=["csv"])
    if not m3_ready:
        st.warning(
            "Model 3 is not trained yet. Run training/train_power_prediction.py after inspecting Dataset 3 and selecting the valid target."
        )
    if uploaded and m3_ready:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)
        if st.button("Predict Expected Power"):
            try:
                predictor = PowerPredictor(
                    MODEL3_DIR / "v1_xgboost_power.pkl",
                    MODEL3_DIR / "feature_columns.json",
                )
                expected = predictor.predict(df)
                st.metric("Mean expected power", f"{expected.mean():.3f}")
                actual_col = st.selectbox(
                    "Actual power column (optional)",
                    ["None"] + list(df.columns),
                )
                if actual_col != "None":
                    deviation = performance_deviation(
                        expected,
                        df[actual_col],
                    )
                    result = pd.DataFrame(
                        {
                            "Expected Power": expected,
                            "Actual Power": df[actual_col],
                            "Deviation %": deviation,
                        }
                    )
                    st.dataframe(result.head(100), use_container_width=True)
                    st.plotly_chart(
                        px.line(
                            result,
                            y=["Expected Power", "Actual Power"],
                            title="Actual vs Expected Power",
                        ),
                        use_container_width=True,
                    )
                    st.metric(
                        "Mean absolute deviation",
                        f"{deviation.abs().mean():.2f}%",
                    )
            except Exception as exc:
                st.error(str(exc))


elif page == "Alert Center":
    st.subheader("🚨 Alert Center")
    st.info(
        "Model 1 alerts are generated from the calibrated anomaly detections. The broader risk engine can later combine Model 1, Model 2 and Model 3 signals. It does not control physical equipment."
    )

    latest = st.session_state.last_fault_result
    if latest:
        alert = latest["alert"]
        if alert["level"] == "CRITICAL":
            st.error(f'🚨 {alert["label"]}')
        elif alert["level"] == "HIGH_RISK":
            st.warning(f'🟠 {alert["label"]}')
        elif alert["level"] == "ATTENTION":
            st.info(f'🟡 {alert["label"]}')
        else:
            st.success("🟢 No trained anomaly detected")

        st.metric("Model 1 severity score", f'{alert["severity_score"]:.0f}/100')
        st.markdown("**Detected:** " + (
            ", ".join(a["name"] for a in latest["detected_anomalies"])
            if latest["detected_anomalies"]
            else "None"
        ))
        st.write(alert["recommendation"])
    else:
        st.info("Run Model 1 from the Fault Detection page to create an alert.")

    st.divider()
    st.markdown("### Combined risk engine — prototype inputs")
    col1, col2, col3 = st.columns(3)
    fault = col1.slider("Fault severity", 0.0, 100.0, 0.0)
    confidence = col2.slider("Fault confidence", 0.0, 100.0, 0.0)
    degradation = col3.number_input("Degradation %", 0.0, 100.0, 0.0)
    deviation = st.number_input("Power deviation %", -100.0, 100.0, 0.0)
    signals = AssetSignals(confidence, fault, degradation, deviation)
    risk = calculate_risk(signals)
    health = health_score(risk)
    level = alert_level(risk)
    st.metric("Combined Risk Score", f"{risk}/100")
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
    st.info(
        f"Digital Twin profile for {asset_id}. Connect the asset registry and saved Model 1/2/3 outputs to populate live state, history and maintenance information."
    )


elif page == "Model Testing Lab":
    st.subheader("🧪 Model Testing Lab")
    st.write(
        "All external testing uses saved models without retraining. Ground truth is required for quantitative validation."
    )
    st.markdown(
        "**Model 1:** external PV image folders are interpreted as multi-label annotations by collapsing exact duplicate hashes and unioning anomaly labels. Calibrated per-anomaly thresholds are used during inference.\n\n"
        "**Model 2:** upload a feature-compatible CSV and run `testing/test_degradation.py`.\n\n"
        "**Model 3:** upload a feature-compatible CSV and run `testing/test_power_prediction.py`."
    )


elif page == "Reports":
    st.subheader("📊 Reports")
    latest = st.session_state.last_fault_result
    if latest:
        st.markdown("### Latest Model 1 report")
        st.json(latest)
    else:
        st.info("Run a Model 1 assessment to populate the report view.")


st.divider()
st.caption(
    "SolarTwin AI — research prototype. PVMD does not contain healthy examples, so Model 1 does not claim validated healthy-vs-defective accuracy. Alert thresholds are prototype inspection-priority rules, not safety standards."
)
