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

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #9aa0aa;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .alert-card {
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        margin: 0.5rem 0 1.2rem 0;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .alert-red { background: rgba(255, 70, 70, 0.12); }
    .alert-orange { background: rgba(255, 150, 0, 0.12); }
    .alert-yellow { background: rgba(255, 210, 0, 0.10); }
    .alert-green { background: rgba(40, 190, 100, 0.10); }
    .small-label {
        color: #9aa0aa;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    return DigitalTwinEngine()


st.markdown('<div class="main-title">☀️ SolarTwin AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">PV Digital Twin for thermal fault detection and power-performance monitoring</div>',
    unsafe_allow_html=True,
)

engine = load_engine()

# -----------------------------------------------------------------------------
# Sidebar: operating information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Panel Information")
    panel_id = st.text_input("Panel ID", "PV-001")
    actual_power = st.number_input(
        "Actual AC Power",
        min_value=0.0,
        value=400.0,
        step=1.0,
        help="Measured AC output for the panel/system at the selected timestamp.",
    )
    irradiation = st.number_input(
        "Irradiation (W/m²)",
        min_value=0.0,
        value=800.0,
        step=10.0,
    )
    ambient = st.number_input("Ambient Temperature (°C)", value=30.0, step=0.5)
    module_temp = st.number_input("Module Temperature (°C)", value=45.0, step=0.5)
    timestamp = st.text_input("Timestamp", "2026-08-11 12:00:00")

    st.divider()
    st.caption("Model 1: visual multi-label anomaly detection")
    st.caption("Model 3: expected AC power prediction")
    st.caption("Model 2: experimental secondary signal")

# -----------------------------------------------------------------------------
# Image upload
# -----------------------------------------------------------------------------
st.subheader("📤 Upload PV Panel Image")
uploaded = st.file_uploader(
    "Upload a thermal / infrared PV panel image",
    type=["jpg", "jpeg", "png"],
    help="The image is passed directly to Model 1 for anomaly detection.",
)

if uploaded is None:
    st.info("Upload a PV panel image to begin the Digital Twin analysis.")
    st.stop()

left, right = st.columns([1.15, 1])
with left:
    st.image(uploaded, caption=uploaded.name, use_container_width=True)

with right:
    st.markdown("### Analysis Inputs")
    st.write(f"**Panel:** {panel_id}")
    st.write(f"**Actual AC Power:** {actual_power:.2f}")
    st.write(f"**Irradiation:** {irradiation:.0f} W/m²")
    st.write(f"**Ambient:** {ambient:.1f} °C")
    st.write(f"**Module:** {module_temp:.1f} °C")
    st.write(f"**Timestamp:** {timestamp}")

    analyze = st.button(
        "🔍 ANALYZE PANEL",
        type="primary",
        use_container_width=True,
    )

if not analyze:
    st.info("Click **ANALYZE PANEL** to run the Digital Twin.")
    st.stop()

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
suffix = Path(uploaded.name).suffix or ".jpg"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    image_path = tmp.name

try:
    with st.spinner("Running Model 1 + Model 3 and building the Digital Twin decision..."):
        result = engine.analyze(
            panel_id=panel_id,
            image_path=image_path,
            power=actual_power,
            timestamp=timestamp,
            ambient_temperature=ambient,
            module_temperature=module_temp,
            irradiation=irradiation,
        )
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

# -----------------------------------------------------------------------------
# Overall decision
# -----------------------------------------------------------------------------
alert_config = {
    "RED": ("🔴 RED ALERT", "Immediate inspection recommended.", "alert-red"),
    "ORANGE": ("🟠 ORANGE ALERT", "Inspection recommended.", "alert-orange"),
    "YELLOW": ("🟡 YELLOW ALERT", "Monitor panel performance closely.", "alert-yellow"),
    "GREEN": ("🟢 GREEN", "No immediate alert from the current models.", "alert-green"),
}
alert_title, recommendation, alert_class = alert_config.get(
    alert,
    ("⚪ UNKNOWN", "Review the analysis output.", "alert-yellow"),
)

st.divider()
st.markdown("## Digital Twin Decision")
st.markdown(
    f'<div class="alert-card {alert_class}"><h2>{alert_title}</h2><p>{recommendation}</p></div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Summary metrics
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Visual Status", visual.get("status", "Unavailable"))
c2.metric("Faults Detected", len(anomalies))
c3.metric("Expected AC Power", f"{expected_power:.2f}")
c4.metric("Power Deviation", f"{deviation:.2f}%")

# -----------------------------------------------------------------------------
# Model 1 results
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Model 3 results
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Maintenance recommendation
# -----------------------------------------------------------------------------
st.divider()
st.markdown("## 🛠️ Maintenance Recommendation")

if alert == "RED":
    st.error(
        "**Priority: Immediate inspection.** Multiple visual anomalies and/or substantial "
        "power underperformance were detected. Inspect the affected PV module and its electrical connections."
    )
elif alert == "ORANGE":
    st.warning(
        "**Priority: Scheduled inspection.** Review the thermal image and power trend and "
        "inspect the module if the condition persists."
    )
elif alert == "YELLOW":
    st.warning(
        "**Priority: Monitor.** Continue tracking thermal anomalies and power deviation."
    )
else:
    st.success("**Priority: Normal.** No immediate maintenance action is indicated by the current models.")

# -----------------------------------------------------------------------------
# Advanced output
# -----------------------------------------------------------------------------
with st.expander("Advanced: Raw Digital Twin Result"):
    st.json(result)

st.caption(
    "Model 2 is retained as an experimental secondary signal and is not used automatically "
    "for the primary fault decision because its current supervised evaluation is not sufficiently validated."
)
