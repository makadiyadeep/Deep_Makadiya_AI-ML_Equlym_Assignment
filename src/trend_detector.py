"""Detect significant month-over-month changes by district and indicator."""

import numpy as np
import pandas as pd

INDICATORS = (
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
)
INPUT_COLUMNS = ("month", "district", *INDICATORS)
TREND_COLUMNS = (
    "district",
    "month",
    "indicator",
    "previous_value",
    "current_value",
    "absolute_change",
    "percentage_change",
    "direction",
)


def detect_trends(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """Return significant adjacent-month changes for each district indicator."""
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be a finite non-negative number") from error
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative number")

    missing_columns = [column for column in INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    if df.empty:
        return pd.DataFrame(columns=TREND_COLUMNS)

    data = df.loc[:, INPUT_COLUMNS].copy()
    data["month"] = pd.to_datetime(data["month"], errors="coerce")
    for indicator in INDICATORS:
        data[indicator] = pd.to_numeric(data[indicator], errors="coerce")
    data = data.dropna(subset=["district", "month"])

    trends: list[dict[str, object]] = []
    for district, district_data in data.groupby("district", sort=True):
        district_data = district_data.sort_values("month", kind="stable")

        for indicator in INDICATORS:
            current = district_data[indicator]
            previous = current.shift(1)
            absolute_change = current - previous
            valid_previous = (
                previous.notna() & np.isfinite(previous) & previous.ne(0)
            )
            percentage_change = pd.Series(np.nan, index=current.index, dtype=float)
            percentage_change.loc[valid_previous] = (
                absolute_change.loc[valid_previous]
                .div(previous.loc[valid_previous])
                .mul(100)
            )

            valid_pair = (
                current.notna()
                & np.isfinite(current)
                & valid_previous
            )
            significant = (
                valid_pair
                & percentage_change.abs().ge(threshold)
                & absolute_change.ne(0)
            )

            for index in district_data.index[significant]:
                change = absolute_change.loc[index]
                trends.append(
                    {
                        "district": district,
                        "month": district_data.at[index, "month"],
                        "indicator": indicator,
                        "previous_value": previous.loc[index],
                        "current_value": current.loc[index],
                        "absolute_change": change,
                        "percentage_change": percentage_change.loc[index],
                        "direction": "increase" if change > 0 else "decrease",
                    }
                )

    return pd.DataFrame(trends, columns=TREND_COLUMNS)
