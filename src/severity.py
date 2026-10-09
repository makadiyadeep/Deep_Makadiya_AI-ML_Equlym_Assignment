"""Engineering severity labels; these rules are not clinically validated."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SeverityRules:
    """Configurable cutoffs for trend and outlier insight severity.

    Trends are Low below 15% absolute change, Medium from 15% to below 30%,
    and High at or above 30%. Outlier severity is based on distance past the
    IQR fence: Low up to 0.5 IQR, Medium above 0.5 through 1.5 IQR, and High
    above 1.5 IQR. These are engineering prioritization choices, not medical
    urgency thresholds. Correlation magnitude alone always receives Low
    severity because association does not establish a harmful event.
    """

    trend_medium_threshold_pct: float = 15.0
    trend_high_threshold_pct: float = 30.0
    outlier_medium_distance_iqr: float = 0.5
    outlier_high_distance_iqr: float = 1.5

    def __post_init__(self) -> None:
        thresholds = (
            self.trend_medium_threshold_pct,
            self.trend_high_threshold_pct,
            self.outlier_medium_distance_iqr,
            self.outlier_high_distance_iqr,
        )
        if not all(math.isfinite(value) and value >= 0 for value in thresholds):
            raise ValueError("Severity thresholds must be finite and non-negative")
        if self.trend_medium_threshold_pct >= self.trend_high_threshold_pct:
            raise ValueError("Trend medium threshold must be below the high threshold")
        if self.outlier_medium_distance_iqr >= self.outlier_high_distance_iqr:
            raise ValueError("Outlier medium-distance threshold must be below the high threshold")


DEFAULT_SEVERITY_RULES = SeverityRules()


def classify_trend_severity(
    percentage_change: float,
    rules: SeverityRules = SeverityRules(),
) -> str:
    magnitude = abs(float(percentage_change))
    if not math.isfinite(magnitude):
        raise ValueError("Trend percentage change must be finite")
    if magnitude < rules.trend_medium_threshold_pct:
        return "Low"
    if magnitude < rules.trend_high_threshold_pct:
        return "Medium"
    return "High"


def classify_outlier_severity(
    value: float,
    direction: str,
    lower_bound: float,
    upper_bound: float,
    iqr: float,
    rules: SeverityRules = SeverityRules(),
) -> str:
    """Classify statistical distance only; this is not clinical severity."""
    if direction == "low":
        distance = float(lower_bound) - float(value)
    elif direction == "high":
        distance = float(value) - float(upper_bound)
    else:
        raise ValueError("Outlier direction must be 'low' or 'high'")

    if not math.isfinite(distance) or not math.isfinite(float(iqr)):
        raise ValueError("Outlier value, bounds, and IQR must be finite")
    if distance < 0:
        raise ValueError("Value is not beyond the supplied IQR bound")
    if iqr <= 0:
        return "Low"

    distance_in_iqrs = distance / float(iqr)
    if distance_in_iqrs <= rules.outlier_medium_distance_iqr:
        return "Low"
    if distance_in_iqrs <= rules.outlier_high_distance_iqr:
        return "Medium"
    return "High"


def classify_correlation_severity() -> str:
    """Correlation is informational; its coefficient alone does not imply risk."""
    return "Low"
