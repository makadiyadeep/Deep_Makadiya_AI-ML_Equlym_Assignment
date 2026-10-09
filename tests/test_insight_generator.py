import pandas as pd
import pytest

from src.insight_generator import INSIGHT_COLUMNS, generate_insights
from src.severity import SeverityRules, classify_outlier_severity, classify_trend_severity


def sample_data():
    return pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "district": ["North", "North"],
            "anc_coverage": [100, 81.2],
            "institutional_delivery": [80, 85],
            "immunization": [90, 92],
            "high_risk_cases": [10, 12],
        }
    )


def trend_results():
    return pd.DataFrame(
        [
            {
                "district": "North",
                "month": pd.Timestamp("2025-02-01"),
                "indicator": "anc_coverage",
                "previous_value": 100.0,
                "current_value": 81.2,
                "absolute_change": -18.8,
                "percentage_change": -18.8,
                "direction": "decrease",
            }
        ]
    )


def outlier_results():
    return pd.DataFrame(
        [
            {
                "district": "South",
                "month": pd.Timestamp("2025-03-01"),
                "indicator": "high_risk_cases",
                "value": 35.0,
                "q1": 14.75,
                "q3": 21.0,
                "iqr": 6.25,
                "lower_bound": 5.375,
                "upper_bound": 30.375,
                "outlier_direction": "high",
            }
        ]
    )


def correlation_results():
    relationships = pd.DataFrame(
        [
            {
                "indicator_1": "anc_coverage",
                "indicator_2": "immunization",
                "correlation": 0.82,
                "relationship": "positive",
                "strength": "strong",
            }
        ]
    )
    return pd.DataFrame(), relationships


def generate_all():
    return generate_insights(
        sample_data(),
        trend_results(),
        outlier_results(),
        correlation_results(),
    )


def test_trend_insight_uses_calculated_change_and_direction():
    insights = generate_insights(
        sample_data(), trend_results(), pd.DataFrame(), (pd.DataFrame(), pd.DataFrame())
    )

    trend = insights.iloc[0]
    assert trend["type"] == "trend"
    assert trend["entity"] == "North"
    assert trend["period"] == "2025-02"
    assert trend["metric"] == pytest.approx(81.2)
    assert trend["change"] == pytest.approx(-18.8)
    assert "decreased by 18.80%" in trend["explanation"]


def test_outlier_insight_contains_district_period_value_and_null_change():
    insights = generate_insights(
        sample_data(), pd.DataFrame(), outlier_results(), (pd.DataFrame(), pd.DataFrame())
    )

    outlier = insights.iloc[0]
    assert outlier["type"] == "outlier"
    assert outlier["entity"] == "South"
    assert outlier["period"] == "2025-03"
    assert outlier["metric"] == pytest.approx(35.0)
    assert pd.isna(outlier["change"])
    assert "high outlier" in outlier["explanation"]


def test_correlation_insight_uses_actual_coefficient_and_nonapplicable_nulls():
    insights = generate_insights(
        sample_data(), pd.DataFrame(), pd.DataFrame(), correlation_results()
    )

    correlation = insights.iloc[0]
    assert correlation["type"] == "correlation"
    assert "anc_coverage" in correlation["entity"]
    assert "immunization" in correlation["entity"]
    assert correlation["metric"] == pytest.approx(0.82)
    assert pd.isna(correlation["period"])
    assert pd.isna(correlation["change"])
    assert "Pearson r = 0.820" in correlation["explanation"]
    assert correlation["severity"] == "Low"


def test_severity_cutoffs_are_configurable_and_independent():
    rules = SeverityRules(
        trend_medium_threshold_pct=8,
        trend_high_threshold_pct=15,
        outlier_medium_distance_iqr=0.2,
        outlier_high_distance_iqr=0.8,
    )

    assert classify_trend_severity(7.9, rules) == "Low"
    assert classify_trend_severity(8, rules) == "Medium"
    assert classify_trend_severity(15, rules) == "High"
    assert classify_outlier_severity(8, "low", 10, 20, 2, rules) == "High"


def test_schema_is_consistent_across_insight_types_and_ids_are_deterministic():
    first = generate_all()
    second = generate_all()

    assert list(first.columns) == list(INSIGHT_COLUMNS)
    assert set(first["type"]) == {"trend", "outlier", "correlation"}
    assert first["insight_id"].is_unique
    assert first["insight_id"].tolist() == second["insight_id"].tolist()
    assert first["explanation"].notna().all()


def test_empty_results_return_empty_dataframe_with_standard_schema():
    insights = generate_insights(
        sample_data(),
        pd.DataFrame(),
        pd.DataFrame(),
        (pd.DataFrame(), pd.DataFrame()),
    )

    assert insights.empty
    assert list(insights.columns) == list(INSIGHT_COLUMNS)
