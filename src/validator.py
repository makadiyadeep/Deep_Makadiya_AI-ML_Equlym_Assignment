
import pandas as pd

REQUIRED_COLUMNS = [
    "month",
    "district",
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]

PERCENTAGE_COLUMNS = [
    "anc_coverage",
    "institutional_delivery",
    "immunization",
]

NUMERIC_COLUMNS = PERCENTAGE_COLUMNS + ["high_risk_cases"]


def validate_dataset(file_path: str) -> pd.DataFrame:
    """Load and validate the healthcare CSV dataset."""

    # 1. Load the dataset
    df = pd.read_csv(file_path)

    print("\n--- Dataset Validation ---")

    # 2. Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 3. Keep only the required columns
    df = df[REQUIRED_COLUMNS].copy()

    # 4. Validate and parse months
    df["month"] = pd.to_datetime(
        df["month"], errors="coerce"
    )

    if df["month"].isna().any():
        raise ValueError("Invalid or missing month values found.")

    # 5. Check missing values
    if df.isna().any().any():
        raise ValueError("Missing values found in the dataset.")

    # 6. Validate district names
    if df["district"].dtype != object:
        raise ValueError("District names must be text.")

    df["district"] = df["district"].str.strip()

    if df["district"].eq("").any():
        raise ValueError("Empty district names found.")

    # 7. Convert numeric columns safely
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if df[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError(
            "Invalid or missing numeric values found."
        )

    # 8. Validate percentage ranges
    for column in PERCENTAGE_COLUMNS:
        invalid = ~df[column].between(0, 100)

        if invalid.any():
            raise ValueError(
                f"{column} contains values outside 0-100."
            )

    # 9. Validate high-risk cases
    if (df["high_risk_cases"] < 0).any():
        raise ValueError("High-risk cases cannot be negative.")

    if (df["high_risk_cases"] % 1 != 0).any():
        raise ValueError("High-risk cases must be integers.")

    df["high_risk_cases"] = df["high_risk_cases"].astype(int)

    # 10. Check duplicate district-month records
    duplicates = df.duplicated(
        subset=["district", "month"]
    )

    if duplicates.any():
        raise ValueError(
            f"Found {duplicates.sum()} duplicate district-month records."
        )

    # 11. Sort the dataset
    df = df.sort_values(
        ["district", "month"]
    ).reset_index(drop=True)

    # 12. Print validation report
    print("Validation passed!")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Districts: {df['district'].nunique()}")
    print(f"Months: {df['month'].nunique()}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate records: {df.duplicated(['district', 'month']).sum()}")

    print("\n--- Numerical Summary ---")
    print(df[NUMERIC_COLUMNS].describe())

    print("\n--- First 5 Rows ---")
    print(df.head())

    return df


if __name__ == "__main__":
    validate_dataset("data/healthcare.csv")
