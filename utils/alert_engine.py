"""Application-level alert and health scoring.

The three model outputs are combined here; the training datasets are never merged.
Thresholds are configurable prototype values, not universal engineering standards.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlertResult:
    risk_score: float
    health_score: float
    level: str
    recommendation: str


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, float(x)))


def compute_alert(
    fault_risk: float = 0.0,
    degradation_risk: float = 0.0,
    power_deviation: float = 0.0,
    w_fault: float = 0.30,
    w_degradation: float = 0.40,
    w_power: float = 0.30,
) -> AlertResult:
    power_risk = _clamp(abs(power_deviation) * 2.0)
    risk = _clamp(w_fault * _clamp(fault_risk) + w_degradation * _clamp(degradation_risk) + w_power * power_risk)
    health = _clamp(100.0 - risk)
    if risk < 20:
        level, recommendation = "NORMAL", "Continue routine monitoring."
    elif risk < 45:
        level, recommendation = "WARNING", "Monitor the asset and review trends."
    elif risk < 70:
        level, recommendation = "HIGH RISK", "Schedule inspection and maintenance review."
    else:
        level, recommendation = "CRITICAL", "Immediate inspection recommended; follow site safety procedures."
    return AlertResult(round(risk, 2), round(health, 2), level, recommendation)
