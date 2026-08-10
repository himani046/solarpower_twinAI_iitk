from dataclasses import dataclass


@dataclass
class AssetSignals:
    fault_confidence: float = 0.0
    fault_severity: float = 0.0
    degradation_pct: float = 0.0
    power_deviation_pct: float = 0.0


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def calculate_risk(signals: AssetSignals) -> float:
    fault_risk = _clamp(signals.fault_severity * signals.fault_confidence / 100.0)
    degradation_risk = _clamp(abs(signals.degradation_pct) / 30.0 * 100.0)
    deviation_risk = _clamp(abs(signals.power_deviation_pct) / 30.0 * 100.0)
    return round(0.30 * fault_risk + 0.40 * degradation_risk + 0.30 * deviation_risk, 2)


def health_score(risk_score: float) -> float:
    return round(100.0 - _clamp(risk_score), 2)


def alert_level(risk_score: float) -> str:
    if risk_score < 20:
        return "NORMAL"
    if risk_score < 45:
        return "WARNING"
    if risk_score < 70:
        return "HIGH RISK"
    return "CRITICAL"


def recommendation(level: str) -> str:
    return {
        "NORMAL": "Continue routine monitoring.",
        "WARNING": "Monitor the asset and review during the next inspection.",
        "HIGH RISK": "Schedule an engineering inspection and maintenance review.",
        "CRITICAL": "Immediate inspection is recommended according to site procedures.",
    }.get(level, "Review the asset according to site procedures.")
