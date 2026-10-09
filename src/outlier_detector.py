"""Detect numerical outliers using an independent IQR per indicator."""

import numpy as np
import pandas as pd

OUTLIER_INDICATORS = (
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
)
REQUIRED_COLUMNS = ("district", "month", *OUTLIER_INDICATORS)
OUTLIER_COLUMNS = (
    "district",
    "month",
    "indicator",
    "value",
    "q1",
    "q3",
    "iqr",
    "lower_bound",
    "upper_bound",
    "outlier_direction",
)
IQR_MULTIPLIER = 1.5


def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """Return outliers with per-indicator IQR bounds.

    Each indicator's quartiles are calculated independently across all finite
    numeric values in the dataset. Missing, non-numeric, and non-finite values
    are excluded from that indicator's quartiles. Outlier measurements without
    a district or month are omitted because they cannot be identified in the
    result.
    """
    if not isinstance(method, str) or method.strip().lower() != "iqr":
        raise ValueError("Unsupported outlier method; only method='iqr' is available")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    if df.empty:
        return pd.DataFrame(columns=OUTLIER_COLUMNS)

    identifiers = df.loc[:, ["district", "month"]].reset_index(drop=True)
    detected: list[dict[str, object]] = []

    for indicator in OUTLIER_INDICATORS:
        values = pd.to_numeric(df[indicator], errors="coerce").astype(float).reset_index(drop=True)
        finite_mask = values.notna() & np.isfinite(values.to_numpy())
        valid_values = values.loc[finite_mask]
        if valid_values.empty:
            continue

        q1 = float(valid_values.quantile(0.25))
        q3 = float(valid_values.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        outlier_mask = finite_mask & ((values < lower_bound) | (values > upper_bound))

        for position in np.flatnonzero(outlier_mask.to_numpy()):
            district = identifiers.at[position, "district"]
            month = identifiers.at[position, "month"]
            if pd.isna(district) or pd.isna(month):
                continue
            value = float(values.iloc[position])
            detected.append(
                {
                    "district": district,
                    "month": month,
                    "indicator": indicator,
                    "value": value,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_direction": "low" if value < lower_bound else "high",
                }
            )

    return pd.DataFrame(detected, columns=OUTLIER_COLUMNS)
