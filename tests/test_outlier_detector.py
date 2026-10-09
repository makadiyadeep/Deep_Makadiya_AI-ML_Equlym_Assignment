import numpy as np
import pandas as pd

from src.outlier_detector import OUTLIER_COLUMNS, detect_outliers


def make_frame(anc_values, immunization_values=None):
    row_count = len(anc_values)
    if immunization_values is None:
        immunization_values = [50] * row_count
    return pd.DataFrame(
        {
            "district": [f"District {index % 2}" for index in range(row_count)],
            "month": pd.date_range("2025-01-01", periods=row_count, freq="MS"),
            "anc_coverage": anc_values,
            "institutional_delivery": [60] * row_count,
            "immunization": immunization_values,
            "high_risk_cases": [10] * row_count,
        }
    )


def test_detects_clear_low_outlier():
    data = make_frame([10, 20, 21, 22, 23, 24, 25, 26])

    outliers = detect_outliers(data)

    assert len(outliers) == 1
    result = outliers.iloc[0]
    assert result["indicator"] == "anc_coverage"
    assert result["value"] == 10
    assert result["outlier_direction"] == "low"
    assert result["value"] < result["lower_bound"]


def test_detects_clear_high_outlier():
    data = make_frame([20, 21, 22, 23, 24, 25, 26, 40])

    outliers = detect_outliers(data)

    assert len(outliers) == 1
    result = outliers.iloc[0]
    assert result["indicator"] == "anc_coverage"
    assert result["value"] == 40
    assert result["outlier_direction"] == "high"
    assert result["value"] > result["upper_bound"]


def test_returns_empty_when_no_outliers_exist():
    data = make_frame([20, 21, 22, 23, 24, 25, 26, 27])

    outliers = detect_outliers(data)

    assert outliers.empty
    assert list(outliers.columns) == list(OUTLIER_COLUMNS)


def test_calculates_bounds_independently_for_each_indicator():
    data = make_frame(
        [10, 20, 21, 22, 23, 24, 25, 26],
        immunization_values=[20, 21, 22, 23, 24, 25, 26, 40],
    )

    outliers = detect_outliers(data)

    assert set(outliers["indicator"]) == {"anc_coverage", "immunization"}
    assert set(outliers["outlier_direction"]) == {"low", "high"}
    assert outliers.set_index("indicator").loc["anc_coverage", "q1"] != outliers.set_index(
        "indicator"
    ).loc["immunization", "q1"]


def test_empty_input_returns_expected_columns():
    columns = [
        "district",
        "month",
        "anc_coverage",
        "institutional_delivery",
        "immunization",
        "high_risk_cases",
    ]

    outliers = detect_outliers(pd.DataFrame(columns=columns))

    assert outliers.empty
    assert list(outliers.columns) == list(OUTLIER_COLUMNS)


def test_missing_invalid_and_nonfinite_values_are_excluded():
    data = make_frame([20, 21, 22, 23, 24, 25, 26, 27, "invalid", None, np.inf])

    outliers = detect_outliers(data)

    assert outliers.empty
