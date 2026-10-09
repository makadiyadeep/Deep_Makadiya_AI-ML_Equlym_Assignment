import pandas as pd
import pytest

from src.trend_detector import TREND_COLUMNS, detect_trends


def observation(month, district="North", **values):
    row = {
        "month": month,
        "district": district,
        "anc_coverage": 80,
        "institutional_delivery": 85,
        "immunization": 90,
        "high_risk_cases": 10,
    }
    row.update(values)
    return row


def test_detects_significant_decrease():
    data = pd.DataFrame(
        [
            observation("2025-01-01", anc_coverage=85),
            observation("2025-02-01", anc_coverage=75),
        ]
    )

    trends = detect_trends(data)

    assert len(trends) == 1
    trend = trends.iloc[0]
    assert trend["indicator"] == "anc_coverage"
    assert trend["absolute_change"] == -10
    assert trend["percentage_change"] == pytest.approx(-100 * 10 / 85)
    assert trend["direction"] == "decrease"


def test_detects_significant_increase_at_threshold():
    data = pd.DataFrame(
        [
            observation("2025-01-01", institutional_delivery=80),
            observation("2025-02-01", institutional_delivery=88),
        ]
    )

    trends = detect_trends(data, threshold=10.0)

    assert len(trends) == 1
    assert trends.iloc[0]["absolute_change"] == 8
    assert trends.iloc[0]["percentage_change"] == pytest.approx(10.0)
    assert trends.iloc[0]["direction"] == "increase"


def test_ignores_change_below_threshold():
    data = pd.DataFrame(
        [
            observation("2025-01-01", immunization=80),
            observation("2025-02-01", immunization=87),
        ]
    )

    assert detect_trends(data).empty


def test_districts_are_analyzed_independently():
    data = pd.DataFrame(
        [
            observation("2025-01-01", "North", anc_coverage=90),
            observation("2025-02-01", "North", anc_coverage=94),
            observation("2025-01-01", "South", anc_coverage=40),
            observation("2025-02-01", "South", anc_coverage=36),
        ]
    )

    trends = detect_trends(data)

    assert len(trends) == 1
    assert trends.iloc[0]["district"] == "South"
    assert trends.iloc[0]["direction"] == "decrease"


def test_indicators_are_analyzed_independently():
    data = pd.DataFrame(
        [
            observation(
                "2025-01-01",
                anc_coverage=80,
                immunization=70,
                high_risk_cases=10,
            ),
            observation(
                "2025-02-01",
                anc_coverage=70,
                immunization=77,
                high_risk_cases=11,
            ),
        ]
    )

    trends = detect_trends(data)

    assert set(trends["indicator"]) == {
        "anc_coverage",
        "immunization",
        "high_risk_cases",
    }
    assert len(trends) == 3


def test_skips_percentage_change_when_previous_value_is_zero():
    data = pd.DataFrame(
        [
            observation("2025-01-01", anc_coverage=0),
            observation("2025-02-01", anc_coverage=80),
        ]
    )

    assert detect_trends(data).empty


def test_empty_dataset_returns_empty_dataframe_with_expected_columns():
    columns = [
        "month",
        "district",
        "anc_coverage",
        "institutional_delivery",
        "immunization",
        "high_risk_cases",
    ]

    trends = detect_trends(pd.DataFrame(columns=columns))

    assert trends.empty
    assert list(trends.columns) == list(TREND_COLUMNS)


def test_single_month_does_not_produce_trends():
    data = pd.DataFrame([observation("2025-01-01", anc_coverage=80)])

    assert detect_trends(data).empty


def test_invalid_observation_breaks_adjacent_comparison_safely():
    data = pd.DataFrame(
        [
            observation("2025-01-01", anc_coverage=80),
            observation("2025-02-01", anc_coverage="invalid"),
            observation("2025-03-01", anc_coverage=100),
        ]
    )

    assert detect_trends(data).empty
