
from pathlib import Path

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
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "healthcare.csv"


def _print_report(checks: list[tuple[str, bool | None, str]]) -> None:
    print("\n--- Dataset Validation Report ---")
    for name, passed, details in checks:
        status = "SKIP" if passed is None else "PASS" if bool(passed) else "FAIL"
        print(f"[{status}] {name}: {details}")

    failures = [name for name, passed, _ in checks if passed is not None and not bool(passed)]
    print(f"\nOverall: {'FAILED' if failures else 'PASSED'}")
    if failures:
        raise ValueError("Validation failed: " + "; ".join(failures))


def validate_dataset(file_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load, convert, and validate the healthcare CSV without changing it."""
    try:
        df = pd.read_csv(file_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        print(f"[FAIL] Load dataset: {error}")
        raise ValueError(f"Could not load dataset at {file_path}") from error

    checks: list[tuple[str, bool | None, str]] = [
        ("Load dataset", True, f"Loaded {len(df)} rows from {file_path}")
    ]
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    checks.append((
        "Required columns",
        not missing_columns,
        "All required columns are present"
        if not missing_columns
        else f"Missing columns: {missing_columns}",
    ))

    if missing_columns:
        for name in (
            "Missing values",
            "Duplicate district-month records",
            "Month datetime conversion",
            "Numeric conversion",
            "Coverage ranges",
            "High-risk case values",
        ):
            checks.append((name, None, "Skipped because required columns are missing"))
        _print_report(checks)

    missing_count = int(df.isna().sum().sum())
    checks.append((
        "Missing values",
        missing_count == 0,
        "No missing values found" if missing_count == 0 else f"Found {missing_count} missing values",
    ))

    parsed_months = pd.to_datetime(df["month"], errors="coerce")
    invalid_month_count = int((df["month"].notna() & parsed_months.isna()).sum())
    checks.append((
        "Month datetime conversion",
        invalid_month_count == 0 and parsed_months.notna().all(),
        "All month values converted to datetime"
        if invalid_month_count == 0 and parsed_months.notna().all()
        else f"Found {invalid_month_count} invalid month values",
    ))
    df["month"] = parsed_months

    duplicate_count = int(df.duplicated(subset=["district", "month"]).sum())
    checks.append((
        "Duplicate district-month records",
        duplicate_count == 0,
        "No duplicate district-month records found"
        if duplicate_count == 0
        else f"Found {duplicate_count} duplicate records",
    ))

    invalid_numeric_counts: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        original = df[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_numeric_counts[column] = int((original.notna() & converted.isna()).sum())
        df[column] = converted
    invalid_numeric_total = sum(invalid_numeric_counts.values())
    checks.append((
        "Numeric conversion",
        invalid_numeric_total == 0,
        "All four healthcare indicators converted to numeric"
        if invalid_numeric_total == 0
        else f"Invalid numeric values by column: {invalid_numeric_counts}",
    ))

    out_of_range = {
        column: int((df[column].notna() & ~df[column].between(0, 100)).sum())
        for column in PERCENTAGE_COLUMNS
    }
    out_of_range_total = sum(out_of_range.values())
    checks.append((
        "Coverage ranges (0-100)",
        out_of_range_total == 0,
        "All coverage values are between 0 and 100"
        if out_of_range_total == 0
        else f"Out-of-range values by column: {out_of_range}",
    ))

    high_risk = df["high_risk_cases"]
    invalid_high_risk = high_risk.notna() & (
        (high_risk < 0) | (high_risk % 1 != 0)
    )
    invalid_high_risk_count = int(invalid_high_risk.sum())
    checks.append((
        "High-risk cases (non-negative integers)",
        invalid_high_risk_count == 0,
        "All high-risk counts are non-negative integers"
        if invalid_high_risk_count == 0
        else f"Found {invalid_high_risk_count} negative or fractional values",
    ))

    _print_report(checks)
    df["high_risk_cases"] = df["high_risk_cases"].astype(int)
    return df


if __name__ == "__main__":
    try:
        validate_dataset()
    except ValueError as error:
        raise SystemExit(str(error)) from error
