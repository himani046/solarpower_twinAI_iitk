from pathlib import Path
import tempfile
import streamlit as st
from inference.digital_twin_engine import DigitalTwinEngine

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="SolarTwin AI", page_icon="☀️", layout="wide")

st.title("☀️ SolarTwin AI")
st.caption("PV Digital Twin — visual fault detection and power-performance monitoring")

@st.cache_resource
def load_engine():
    return DigitalTwinEngine()

engine = load_engine()

with st.sidebar:
    st.header("Panel Information")
    panel_id = st.text_input("Panel ID", "PV-001")
    actual_power = st.number_input("Actual AC Power", min_value=0.0, value=400.0, step=1.0)
    irradiation = st.number_input("Irradiation (W/m²)", min_value=0.0, value=800.0, step=10.0)
    ambient = st.number_input("Ambient Temperature (°C)", value=30.0, step=0.5)
    module_temp = st.number_input("Module Temperature (°C)", value=45.0, step=0.5)
    timestamp = st.text_input("Timestamp", "2026-08-11 12:00:00")

uploaded = st.file_uploader("Upload PV Panel Thermal / Visual Image", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Upload a PV panel image to begin analysis.")
else:
    st.image(uploaded, caption="Uploaded PV panel", use_container_width=True)
    if st.button("🔍 ANALYZE PANEL", type="primary", use_container_width=True):
        suffix = Path(uploaded.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            image_path = tmp.name
        try:
            with st.spinner("Running SolarTwin AI..."):
                result = engine.analyze(panel_id=panel_id, image_path=image_path, power=actual_power, timestamp=timestamp, ambient_temperature=ambient, module_temperature=module_temp, irradiation=irradiation)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

        visual = result.get("visual_fault") or {}
        power = result.get("power") or {}
        alert = result.get("alert_level", "UNKNOWN")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Visual Status", visual.get("status", "Unavailable"))
        c2.metric("Expected AC Power", f"{power.get('expected_ac_power', 0):.2f}")
        c3.metric("Power Deviation", f"{power.get('deviation_percent', 0):.2f}%")

        if alert == "RED":
            st.error("🔴 RED ALERT — Immediate inspection recommended")
        elif alert == "ORANGE":
            st.warning("🟠 ORANGE — Inspection recommended")
        elif alert == "YELLOW":
            st.warning("🟡 YELLOW — Monitor panel performance")
        else:
            st.success("🟢 GREEN — No immediate alert")

        st.subheader("Visual Fault Detection")
        anomalies = visual.get("anomalies", [])
        probs = visual.get("probabilities", {})
        if anomalies:
            st.write("**Detected anomalies:** " + ", ".join(anomalies))
            for name, probability in probs.items():
                st.progress(float(probability), text=f"{name}: {probability:.2%}")
        else:
            st.success("No trained anomaly label crossed its configured threshold.")

        st.subheader("Power Performance")
        p1, p2, p3 = st.columns(3)
        p1.metric("Actual AC Power", f"{power.get('actual_ac_power', actual_power):.2f}")
        p2.metric("Expected AC Power", f"{power.get('expected_ac_power', 0):.2f}")
        p3.metric("Daylight", "Yes" if power.get("daylight") else "No")

        st.subheader("Digital Twin Decision")
        st.json(result)
        st.caption("Model 2 is not used automatically because its current supervised evaluation is experimental and not sufficiently validated.")
