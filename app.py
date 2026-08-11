from pathlib import Path
import tempfile

import streamlit as st

from inference.digital_twin_engine import DigitalTwinEngine

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SolarTwin AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #F8F7FF; color: #0B1D2D; }
    .main .block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] { background: #0B1D2D; border-right: 1px solid #172D40; }
    section[data-testid="stSidebar"] * { color: #F8F7FF !important; }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea { background: #13293B !important; border: 1px solid #31506A !important; color: #FFFFFF !important; border-radius: 10px !important; }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button { color: #FFFFFF !important; background: #17344A !important; border-color: #31506A !important; }
    section[data-testid="stSidebar"] hr { border-color: #29445A !important; }
    .brand-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.15rem; }
    .brand-icon { width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #12B8B0, #3DA9FC); box-shadow: 0 8px 24px rgba(18,184,176,0.22); font-size: 1.45rem; }
    .main-title { color: #0B1D2D; font-size: 2.55rem; line-height: 1.05; font-weight: 850; letter-spacing: -0.045em; margin: 0; }
    .main-title .accent { color: #E77C83; }
    .subtitle { color: #536577; font-size: 1rem; margin: 0.45rem 0 1.6rem 3.35rem; }
    .eyebrow { display: inline-block; padding: 0.38rem 0.75rem; border-radius: 999px; background: #EFE8FF; color: #5A24BD; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.8rem; }
    h1, h2, h3 { color: #0B1D2D !important; letter-spacing: -0.025em; }
    .soft-card, .input-card { background: #FFFFFF; border: 1px solid #E5E0F0; border-radius: 18px; padding: 1.15rem 1.3rem; box-shadow: 0 8px 28px rgba(11,29,45,0.055); }
    .input-card { min-height: 245px; }
    .small-label, .decision-label { color: #718096; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 800; }
    .decision-label { font-size: 0.78rem; margin-bottom: 0.25rem; }
    .alert-card { padding: 1.15rem 1.35rem; border-radius: 18px; margin: 0.4rem 0 1.2rem 0; border: 1px solid; box-shadow: 0 8px 26px rgba(11,29,45,0.05); }
    .alert-card h2 { margin: 0 0 0.25rem 0; font-size: 1.45rem; }
    .alert-card p { margin: 0; color: #536577; }
    .alert-red { background: #FFF1F2; border-color: #F4B7BC; }
    .alert-orange { background: #FFF5EA; border-color: #F5C58F; }
    .alert-yellow { background: #FFFBEA; border-color: #EAD58A; }
    .alert-green { background: #ECFBF8; border-color: #A8E1D9; }
    .stButton > button { border-radius: 12px !important; border: 0 !important; color: #FFFFFF !important; background: linear-gradient(90deg, #6C2BD9 0%, #8B45E6 55%, #E77C83 100%) !important; font-weight: 800 !important; min-height: 2.8rem; box-shadow: 0 8px 20px rgba(108,43,217,0.20); transition: transform 0.15s ease, box-shadow 0.15s ease; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 11px 24px rgba(108,43,217,0.28); }
    [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E5E0F0; border-radius: 16px; padding: 0.8rem 1rem; box-shadow: 0 6px 20px rgba(11,29,45,0.04); }
    [data-testid="stMetricLabel"] { color: #718096 !important; }
    [data-testid="stMetricValue"] { color: #0B1D2D !important; font-weight: 800; }

    /* File uploader - explicit light-theme button styling */
    [data-testid="stFileUploader"] { background: #FFFFFF; border: 1.5px dashed #C7B8E8; border-radius: 16px; padding: 0.35rem; }
    [data-testid="stFileUploader"] section { background: #FCFBFF; border-radius: 13px; }
    [data-testid="stFileUploader"] button { background: linear-gradient(90deg, #6C2BD9 0%, #8B45E6 55%, #E77C83 100%) !important; color: #FFFFFF !important; border: 0 !important; border-radius: 10px !important; font-weight: 800 !important; box-shadow: 0 6px 16px rgba(108,43,217,0.18) !important; }
    [data-testid="stFileUploader"] button:hover { background: linear-gradient(90deg, #5A24BD 0%, #7836D0 55%, #D96972 100%) !important; color: #FFFFFF !important; }
    [data-testid="stFileUploader"] button span { color: #FFFFFF !important; }
    [data-testid="stFileUploader"] small { color: #718096 !important; }

    [data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #12B8B0, #6C2BD9, #E77C83); }
    hr { border-color: #E5E0F0 !important; }
    [data-testid="stAlert"] { border-radius: 14px; }
    details { background: #FFFFFF; border: 1px solid #E5E0F0 !important; border-radius: 14px !important; }
    a { color: #6C2BD9 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_engine():
    return DigitalTwinEngine()

st.markdown('<div class="eyebrow">☀️ AI-POWERED PV DIGITAL TWIN</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-row"><div class="brand-icon">☀️</div><div class="main-title">SolarTwin <span class="accent">AI</span></div></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Visual fault detection, expected power prediction, and intelligent PV maintenance decisions.</div>', unsafe_allow_html=True)

engine = load_engine()

with st.sidebar:
    st.header("Panel Information")
    panel_id = st.text_input("Panel ID", "PV-001")
    actual_power = st.number_input("Actual AC Power", min_value=0.0, value=400.0, step=1.0, help="Measured AC output for the panel/system at the selected timestamp.")
    irradiation = st.number_input("Irradiation (W/m²)", min_value=0.0, value=800.0, step=10.0)
    ambient = st.number_input("Ambient Temperature (°C)", value=30.0, step=0.5)
    module_temp = st.number_input("Module Temperature (°C)", value=45.0, step=0.5)
    timestamp = st.text_input("Timestamp", "2026-08-11 12:00:00")
    st.divider()
    st.caption("MODEL 1  •  Visual multi-label anomaly detection")
    st.caption("MODEL 3  •  Expected AC power prediction")
    st.caption("MODEL 2  •  Experimental secondary signal")

st.markdown("## Upload & Analyze")
uploaded = st.file_uploader("Upload a thermal / infrared PV panel image", type=["jpg", "jpeg", "png"], help="The uploaded image is passed directly to Model 1 for anomaly detection.")

if uploaded is None:
    st.info("📤 Upload a PV panel image to begin the Digital Twin analysis.")
    st.stop()

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.markdown('<div class="small-label">Thermal / Visual Input</div>', unsafe_allow_html=True)
    st.image(uploaded, caption=uploaded.name, use_container_width=True)
with right:
    st.markdown('<div class="small-label">Analysis Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown(f"**Panel ID:** {panel_id}")
    st.markdown(f"**Actual AC Power:** {actual_power:.2f}")
    st.markdown(f"**Irradiation:** {irradiation:.0f} W/m²")
    st.markdown(f"**Ambient:** {ambient:.1f} °C")
    st.markdown(f"**Module:** {module_temp:.1f} °C")
    st.markdown(f"**Timestamp:** {timestamp}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    analyze = st.button("🔍  ANALYZE PANEL", type="primary", use_container_width=True)

if not analyze:
    st.info("Click **ANALYZE PANEL** to run the SolarTwin AI Digital Twin.")
    st.stop()

suffix = Path(uploaded.name).suffix or ".jpg"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    image_path = tmp.name

try:
    with st.spinner("Running Model 1 + Model 3 and building the Digital Twin decision..."):
        result = engine.analyze(panel_id=panel_id, image_path=image_path, power=actual_power, timestamp=timestamp, ambient_temperature=ambient, module_temperature=module_temp, irradiation=irradiation)
except Exception as exc:
    st.error(f"Inference failed: {exc}")
    st.stop()

visual = result.get("visual_fault") or {}
power = result.get("power") or {}
alert = str(result.get("alert_level", "UNKNOWN")).upper()
anomalies = visual.get("anomalies", [])
probabilities = visual.get("probabilities", {})
expected_power = float(power.get("expected_ac_power") or 0.0)
actual_ac_power = float(power.get("actual_ac_power") or actual_power)
deviation = float(power.get("deviation_percent") or 0.0)

alert_config = {
    "RED": ("🔴 RED ALERT", "Immediate inspection recommended.", "alert-red"),
    "ORANGE": ("🟠 ORANGE ALERT", "Inspection recommended.", "alert-orange"),
    "YELLOW": ("🟡 YELLOW ALERT", "Monitor panel performance closely.", "alert-yellow"),
    "GREEN": ("🟢 GREEN", "No immediate alert from the current models.", "alert-green"),
}
alert_title, recommendation, alert_class = alert_config.get(alert, ("⚪ UNKNOWN", "Review the analysis output.", "alert-yellow"))

st.divider()
st.markdown("## Digital Twin Decision")
st.markdown(f'<div class="alert-card {alert_class}"><h2>{alert_title}</h2><p>{recommendation}</p></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Visual Status", visual.get("status", "Unavailable"))
c2.metric("Faults Detected", len(anomalies))
c3.metric("Expected AC Power", f"{expected_power:.2f}")
c4.metric("Power Deviation", f"{deviation:.2f}%")

st.divider()
st.markdown("## 🔍 Visual Fault Detection")
if anomalies:
    st.error("Multiple anomalies detected: " + ", ".join(anomalies))
else:
    st.success("No trained anomaly label crossed its configured threshold.")

fault_cols = st.columns(max(1, len(probabilities)))
for idx, (name, probability) in enumerate(probabilities.items()):
    with fault_cols[idx % len(fault_cols)]:
        probability = float(probability)
        st.metric(name, f"{probability:.2%}")
        st.progress(min(max(probability, 0.0), 1.0))

st.divider()
st.markdown("## ⚡ Power Performance")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Actual AC Power", f"{actual_ac_power:.2f}")
p2.metric("Expected AC Power", f"{expected_power:.2f}")
p3.metric("Deviation", f"{deviation:.2f}%")
p4.metric("Daylight", "Yes" if power.get("daylight") else "No")

if expected_power > 0:
    if deviation >= 30:
        st.error("Power output is substantially below the model's expected output.")
    elif deviation >= 15:
        st.warning("Power output is below the model's expected output.")
    else:
        st.success("Power output is close to the model's expected output.")

st.divider()
st.markdown("## 🛠️ Maintenance Recommendation")
if alert == "RED":
    st.error("**Priority: Immediate inspection.** Multiple visual anomalies and/or substantial power underperformance were detected. Inspect the affected PV module and its electrical connections.")
elif alert == "ORANGE":
    st.warning("**Priority: Scheduled inspection.** Review the thermal image and power trend and inspect the module if the condition persists.")
elif alert == "YELLOW":
    st.warning("**Priority: Monitor.** Continue tracking thermal anomalies and power deviation.")
else:
    st.success("**Priority: Normal.** No immediate maintenance action is indicated by the current models.")

with st.expander("Advanced: Raw Digital Twin Result"):
    st.json(result)

st.caption("Model 2 is retained as an experimental secondary signal and is not used automatically for the primary fault decision because its current supervised evaluation is not sufficiently validated.")
