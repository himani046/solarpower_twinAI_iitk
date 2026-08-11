"""Worker-facing alert logic for Model 1 PV anomaly detection.

This is an application-layer prototype, not a safety certification or
engineering protection system. It converts multi-label anomaly predictions
into a simple inspection priority while preserving the anomaly names and
confidence values.
"""
from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def build_model1_alert(detected_anomalies: list[dict]) -> dict:
    """Return status, priority and worker guidance from Model 1 detections."""
    anomalies = sorted(
        detected_anomalies,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )

    if not anomalies:
        return {
            "level": "NO_TRAINED_ANOMALY",
            "label": "No trained anomaly detected",
            "severity_score": 0.0,
            "recommendation": "No trained anomaly crossed its calibrated threshold. Do not label the panel healthy; continue routine monitoring or inspect if other signals disagree.",
        }

    max_confidence = _clamp(anomalies[0]["confidence"])
    count = len(anomalies)

    # Prototype inspection-priority rules. These are intentionally separate
    # from model thresholds: a low calibrated detection threshold can identify
    # a possible anomaly without immediately creating a critical alert.
    if (count >= 2 and max_confidence >= 80.0) or max_confidence >= 95.0:
        level = "CRITICAL"
        label = "Critical inspection required"
        recommendation = "Multiple/high-confidence anomalies detected. Prioritize field inspection, isolate the affected asset if required by site procedures, and follow electrical safety procedures."
    elif count >= 2 or max_confidence >= 60.0:
        level = "HIGH_RISK"
        label = "High-risk anomaly detected"
        recommendation = "Prioritize inspection and maintenance review. Check the affected module/string and correlate with electrical and power-performance signals."
    else:
        level = "ATTENTION"
        label = "Attention required"
        recommendation = "Review the image and schedule an inspection. Correlate with electrical and operational data before taking maintenance action."

    severity_score = _clamp(
        max_confidence + (10.0 if count >= 2 else 0.0)
    )

    return {
        "level": level,
        "label": label,
        "severity_score": round(severity_score, 2),
        "recommendation": recommendation,
    }
