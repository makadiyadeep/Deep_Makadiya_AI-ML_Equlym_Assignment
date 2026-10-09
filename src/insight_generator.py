"""Convert existing analytical results into a consistent insight schema.

Insight fields: ``insight_id`` is a deterministic per-run sequence; ``type``
identifies trend/outlier/correlation; ``indicator`` identifies one metric or a
pair; ``entity`` is a district or the correlation pair; ``period`` is the
relevant month (null for correlations); ``metric`` is the observed value or
coefficient; ``change`` is a trend percentage (null when inapplicable);
``severity`` is an engineering priority label; ``explanation`` is generated
from the supplied analytical results. Severity cutoffs are not medically
validated.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.correlation_analyzer import INDICATOR_LABELS
from src.correlation_analyzer import analyze_correlations
from src.outlier_detector import detect_outliers
from src.severity import (
    DEFAULT_SEVERITY_RULES,
    SeverityRules,
    classify_correlation_severity,
    classify_outlier_severity,
    classify_trend_severity,
)
from src.trend_detector import detect_trends
from src.validator import DEFAULT_DATA_PATH, validate_dataset

INSIGHT_COLUMNS = (
    "insight_id",
    "type",
    "indicator",
    "entity",
    "period",
    "metric",
    "change",
    "severity",
    "explanation",
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _label(indicator: object) -> str:
    name = str(indicator)
    return INDICATOR_LABELS.get(name, name.replace("_", " ").title())


def _period(value: object) -> str | None:
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.strftime("%Y-%m") if not pd.isna(parsed) else str(value)


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _result_frame(
    result: object,
    required_columns: tuple[str, ...],
    name: str,
) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame(columns=required_columns)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"{name} results must be a pandas DataFrame")
    missing = [column for column in required_columns if column not in result.columns]
    if missing and not result.empty:
        raise ValueError(f"{name} results are missing columns: {missing}")
    return result.copy() if not missing else pd.DataFrame(columns=required_columns)


def _correlation_frame(result: object) -> pd.DataFrame:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("Correlation results must be (matrix, relationships)")
        result = result[1]
    return _result_frame(
        result,
        ("indicator_1", "indicator_2", "correlation", "relationship", "strength"),
        "Correlation",
    )


def _ordered(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame
    available = [column for column in columns if column in frame.columns]
    return frame.sort_values(available, kind="stable", na_position="last") if available else frame


def generate_insights(
    df: pd.DataFrame,
    trend_results: pd.DataFrame,
    outlier_results: pd.DataFrame,
    correlation_results: tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame,
    *,
    severity_rules: SeverityRules = DEFAULT_SEVERITY_RULES,
) -> pd.DataFrame:
    """Build deterministic insight records from precomputed analysis results.

    ``correlation_results`` may be the pair returned by
    ``analyze_correlations`` or its relationships DataFrame. Calculations are
    not repeated here; malformed/non-finite result rows are skipped rather
    than assigned fabricated values.
    """
    trends = _result_frame(
        trend_results,
        ("district", "month", "indicator", "current_value", "percentage_change", "direction"),
        "Trend",
    )
    outliers = _result_frame(
        outlier_results,
        (
            "district", "month", "indicator", "value", "iqr", "lower_bound",
            "upper_bound", "outlier_direction",
        ),
        "Outlier",
    )
    correlations = _correlation_frame(correlation_results)
    insights: list[dict[str, object]] = []

    for _, row in _ordered(trends, ("district", "month", "indicator")).iterrows():
        change = _finite_number(row["percentage_change"])
        metric = _finite_number(row["current_value"])
        if change is None or metric is None or pd.isna(row["district"]):
            continue
        direction = str(row["direction"]).lower()
        if direction not in {"increase", "decrease"}:
            continue
        indicator = str(row["indicator"])
        period = _period(row["month"])
        label = _label(indicator)
        severity = classify_trend_severity(change, severity_rules)
        explanation = (
            f"{label} in {row['district']} {direction}d by {abs(change):.2f}% "
            f"compared with the previous month ({period}); current value: {metric:g}."
        )
        insights.append(
            {
                "type": "trend",
                "indicator": indicator,
                "entity": str(row["district"]),
                "period": period,
                "metric": metric,
                "change": change,
                "severity": severity,
                "explanation": explanation,
            }
        )

    for _, row in _ordered(outliers, ("district", "month", "indicator")).iterrows():
        value = _finite_number(row["value"])
        iqr = _finite_number(row["iqr"])
        lower = _finite_number(row["lower_bound"])
        upper = _finite_number(row["upper_bound"])
        if (
            value is None or iqr is None or lower is None or upper is None
            or pd.isna(row["district"])
        ):
            continue
        direction = str(row["outlier_direction"]).lower()
        if direction not in {"low", "high"}:
            continue
        indicator = str(row["indicator"])
        period = _period(row["month"])
        label = _label(indicator)
        severity = classify_outlier_severity(value, direction, lower, upper, iqr, severity_rules)
        bound = lower if direction == "low" else upper
        distance = lower - value if direction == "low" else value - upper
        unit = "cases" if indicator == "high_risk_cases" else "percentage points"
        explanation = (
            f"{label} in {row['district']} was flagged as a {direction} outlier "
            f"relative to the observed distribution ({period}; value {value:g}, "
            f"IQR bound {bound:.2f}; {distance:.2f} {unit} beyond the bound). "
            "A statistical outlier is not by itself a clinical emergency."
        )
        insights.append(
            {
                "type": "outlier",
                "indicator": indicator,
                "entity": str(row["district"]),
                "period": period,
                "metric": value,
                "change": None,
                "severity": severity,
                "explanation": explanation,
            }
        )

    for _, row in _ordered(correlations, ("indicator_1", "indicator_2")).iterrows():
        coefficient = _finite_number(row["correlation"])
        if coefficient is None:
            continue
        indicator_1 = str(row["indicator_1"])
        indicator_2 = str(row["indicator_2"])
        relationship = str(row["relationship"]).lower()
        strength = str(row["strength"]).lower()
        first_label = _label(indicator_1)
        second_label = _label(indicator_2)
        pair = f"{indicator_1} & {indicator_2}"
        explanation = (
            f"{first_label} and {second_label} showed a {strength} {relationship} "
            f"linear association (Pearson r = {coefficient:.3f}) in the observed "
            f"dataset of {len(df)} rows. Correlation indicates association, not "
            "causation; these results use synthetic data."
        )
        insights.append(
            {
                "type": "correlation",
                "indicator": f"{indicator_1}, {indicator_2}",
                "entity": pair,
                "period": None,
                "metric": coefficient,
                "change": None,
                "severity": classify_correlation_severity(),
                "explanation": explanation,
            }
        )

    for insight_id, insight in enumerate(insights, start=1):
        insight["insight_id"] = f"insight-{insight_id:04d}"
    return pd.DataFrame(insights, columns=INSIGHT_COLUMNS)


def run_workflow(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    trend_threshold: float = 10.0,
    correlation_threshold: float = 0.70,
    severity_rules: SeverityRules = DEFAULT_SEVERITY_RULES,
) -> pd.DataFrame:
    """Run existing analyses, generate insights, and save CSV and JSON outputs."""
    data = validate_dataset(data_path)
    trend_results = detect_trends(data, threshold=trend_threshold)
    outlier_results = detect_outliers(data)
    correlation_results = analyze_correlations(data, threshold=correlation_threshold)
    insights = generate_insights(
        data,
        trend_results,
        outlier_results,
        correlation_results,
        severity_rules=severity_rules,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "insights.csv"
    json_path = output_path / "sample_insights.json"
    insights.to_csv(csv_path, index=False)
    json_path.write_text(insights.to_json(orient="records", indent=2), encoding="utf-8")

    print("\n--- Generated Insight Summary ---")
    print(f"Total insights: {len(insights)}")
    print("Count by type:")
    print(insights["type"].value_counts().reindex(["trend", "outlier", "correlation"], fill_value=0).to_string())
    print("Count by severity:")
    print(insights["severity"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).to_string())
    print("Sample explanations:")
    for explanation in insights["explanation"].head(5):
        print(f"- {explanation}")
    print(f"Insights CSV saved: {csv_path}")
    print(f"Insights JSON saved: {json_path}")
    return insights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Healthcare CSV path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for insight outputs")
    parser.add_argument("--trend-threshold", type=float, default=10.0, help="Minimum absolute month-over-month change in percent")
    parser.add_argument("--correlation-threshold", type=float, default=0.70, help="Minimum absolute Pearson coefficient")
    parser.add_argument("--trend-medium-severity", type=float, default=15.0, help="Trend percentage cutoff for Medium severity")
    parser.add_argument("--trend-high-severity", type=float, default=30.0, help="Trend percentage cutoff for High severity")
    parser.add_argument("--outlier-medium-iqr", type=float, default=0.5, help="Outlier distance in IQR units for Medium severity")
    parser.add_argument("--outlier-high-iqr", type=float, default=1.5, help="Outlier distance in IQR units for High severity")
    arguments = parser.parse_args()
    rules = SeverityRules(
        trend_medium_threshold_pct=arguments.trend_medium_severity,
        trend_high_threshold_pct=arguments.trend_high_severity,
        outlier_medium_distance_iqr=arguments.outlier_medium_iqr,
        outlier_high_distance_iqr=arguments.outlier_high_iqr,
    )
    run_workflow(
        data_path=arguments.data,
        output_dir=arguments.output_dir,
        trend_threshold=arguments.trend_threshold,
        correlation_threshold=arguments.correlation_threshold,
        severity_rules=rules,
    )


if __name__ == "__main__":
    main()
