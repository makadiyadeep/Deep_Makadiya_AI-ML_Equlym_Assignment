"""Generate a reproducible synthetic district-level healthcare dataset."""

from __future__ import annotations

import csv
import math
import random
from calendar import monthrange
from pathlib import Path
from statistics import mean, stdev

try:
    import pandas as pd
except ImportError:
    pd = None


SEED = 2025
DISTRICTS = (
    "Ahmedabad",
    "Surat",
    "Vadodara",
    "Rajkot",
    "Mehsana",
    "Bhavnagar",
    "Jamnagar",
    "Junagadh",
    "Anand",
    "Kheda",
    "Bharuch",
    "Valsad",
    "Navsari",
    "Gandhinagar",
    "Patan",
    "Amreli",
    "Porbandar",
    "Morbi",
    "Sabarkantha",
    "Panchmahal",
)
MONTHS = tuple(f"2025-{month:02d}-01" for month in range(1, 13))
COLUMNS = (
    "month",
    "district",
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
)
COVERAGE_COLUMNS = ("anc_coverage", "institutional_delivery", "immunization")
NUMERIC_COLUMNS = (*COVERAGE_COLUMNS, "high_risk_cases")

# These are synthetic scenario values, chosen to make trend and IQR checks useful.
SCENARIOS = {
    ("Ahmedabad", "2025-08-01", "anc_coverage"): 86,
    ("Ahmedabad", "2025-09-01", "anc_coverage"): 70,
    ("Ahmedabad", "2025-10-01", "anc_coverage"): 79,
    ("Mehsana", "2025-06-01", "anc_coverage"): 83,
    ("Mehsana", "2025-06-01", "high_risk_cases"): 12,
    ("Mehsana", "2025-07-01", "anc_coverage"): 44,
    ("Mehsana", "2025-07-01", "high_risk_cases"): 35,
    ("Mehsana", "2025-08-01", "anc_coverage"): 80,
    ("Mehsana", "2025-08-01", "high_risk_cases"): 13,
    ("Surat", "2025-03-01", "immunization"): 78,
    ("Surat", "2025-04-01", "immunization"): 89,
    ("Surat", "2025-05-01", "immunization"): 79,
    ("Vadodara", "2025-02-01", "anc_coverage"): 76,
    ("Vadodara", "2025-03-01", "anc_coverage"): 87,
    ("Rajkot", "2025-11-01", "high_risk_cases"): 15,
    ("Rajkot", "2025-12-01", "high_risk_cases"): 10,
    ("Gandhinagar", "2025-03-01", "institutional_delivery"): 79,
    ("Gandhinagar", "2025-04-01", "institutional_delivery"): 89,
    ("Patan", "2025-05-01", "institutional_delivery"): 84,
    ("Patan", "2025-06-01", "institutional_delivery"): 64,
    ("Patan", "2025-07-01", "institutional_delivery"): 82,
    ("Amreli", "2025-10-01", "immunization"): 65,
    ("Panchmahal", "2025-04-01", "anc_coverage"): 56,
    ("Bharuch", "2025-09-01", "institutional_delivery"): 64,
}


def build_rows() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    quality_levels = list(range(-9, 11))
    rng.shuffle(quality_levels)
    rows: list[dict[str, object]] = []

    for district_index, district in enumerate(DISTRICTS):
        quality = quality_levels[district_index]
        trend = (0.42, -0.22, 0.04)[district_index % 3]
        immunization_offset = rng.uniform(-4.0, 4.0)
        risk_offset = rng.gauss(0.0, 1.0)
        previous_noise = [0.0, 0.0, 0.0]

        for month_index, month in enumerate(MONTHS):
            seasonal = 0.55 * math.sin((month_index / 12) * math.tau)
            previous_noise = [
                previous_noise[i] * 0.35 + rng.gauss(0.0, 0.9)
                for i in range(3)
            ]
            progress = trend * month_index + seasonal
            anc = 78 + quality * 0.82 + progress + previous_noise[0]
            delivery = 86 + quality * 0.74 + progress * 0.52 + previous_noise[1]
            immunization = (
                87
                + quality * 0.32
                + immunization_offset
                + progress * 0.24
                + previous_noise[2]
            )
            high_risk = round(
                18 - quality * 0.62 - trend * month_index * 0.3
                + risk_offset
                + rng.gauss(0.0, 1.35)
            )
            row: dict[str, object] = {
                "month": month,
                "district": district,
                "anc_coverage": round(anc, 1),
                "institutional_delivery": round(delivery, 1),
                "immunization": round(immunization, 1),
                "high_risk_cases": max(0, high_risk),
            }
            for column in NUMERIC_COLUMNS:
                if (district, month, column) in SCENARIOS:
                    row[column] = SCENARIOS[(district, month, column)]
            rows.append(row)

    return sorted(rows, key=lambda row: (row["district"], row["month"]))


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def validate(rows: list[dict[str, object]]) -> tuple[int, dict[str, float]]:
    if len(rows) != 240:
        raise ValueError(f"Expected 240 rows, found {len(rows)}")
    if any(tuple(row) != COLUMNS for row in rows):
        raise ValueError("Rows do not match the required column names and order")
    if [row["district"] for row in rows] != sorted(row["district"] for row in rows):
        raise ValueError("Rows are not sorted by district and month")
    if len({row["district"] for row in rows}) != 20:
        raise ValueError("Expected exactly 20 distinct districts")
    if len({row["month"] for row in rows}) != 12:
        raise ValueError("Expected exactly 12 distinct months")
    if any((row["district"], row["month"]) != pair for row, pair in zip(
        rows,
        sorted((row["district"], row["month"]) for row in rows),
    )):
        raise ValueError("Rows are not sorted by district and month")
    if len({(row["district"], row["month"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate district-month combinations found")
    if any(row[column] is None for row in rows for column in COLUMNS):
        raise ValueError("Missing values found")

    for district in DISTRICTS:
        district_rows = [row for row in rows if row["district"] == district]
        if len(district_rows) != 12 or [row["month"] for row in district_rows] != list(MONTHS):
            raise ValueError(f"{district} does not have exactly one row per month")
    for row in rows:
        for column in COVERAGE_COLUMNS:
            value = row[column]
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(f"Invalid {column} value for {row['district']} {row['month']}: {value}")
        count = row["high_risk_cases"]
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid high_risk_cases value for {row['district']} {row['month']}: {count}")

    outlier_counts: dict[str, int] = {}
    outlier_total = 0
    low_outlier_total = 0
    high_outlier_total = 0
    for column in NUMERIC_COLUMNS:
        values = [float(row[column]) for row in rows]
        lower = quantile(values, 0.25) - 1.5 * (quantile(values, 0.75) - quantile(values, 0.25))
        upper = quantile(values, 0.75) + 1.5 * (quantile(values, 0.75) - quantile(values, 0.25))
        low_count = sum(value < lower for value in values)
        high_count = sum(value > upper for value in values)
        count = low_count + high_count
        outlier_counts[column] = count
        outlier_total += count
        low_outlier_total += low_count
        high_outlier_total += high_count
    if outlier_total < 5:
        raise ValueError(f"Expected at least 5 IQR outlier observations, found {outlier_total}")
    if low_outlier_total == 0 or high_outlier_total == 0:
        raise ValueError("Expected both low and high IQR outlier observations")

    changes = 0
    district_values = {
        district: [row for row in rows if row["district"] == district]
        for district in DISTRICTS
    }
    for district_rows in district_values.values():
        for previous, current in zip(district_rows, district_rows[1:]):
            for column in NUMERIC_COLUMNS:
                previous_value = float(previous[column])
                if previous_value > 0 and abs((float(current[column]) - previous_value) / previous_value) >= 0.10:
                    changes += 1
    if changes < 10:
        raise ValueError(f"Expected at least 10 month-to-month changes of 10% or more, found {changes}")

    numeric_data = {
        column: [float(row[column]) for row in rows]
        for column in NUMERIC_COLUMNS
    }
    correlations = {
        "ANC / institutional delivery": pearson(
            numeric_data["anc_coverage"], numeric_data["institutional_delivery"]
        ),
        "ANC / high-risk cases": pearson(
            numeric_data["anc_coverage"], numeric_data["high_risk_cases"]
        ),
        "ANC / immunization": pearson(
            numeric_data["anc_coverage"], numeric_data["immunization"]
        ),
    }
    if correlations["ANC / institutional delivery"] < 0.70:
        raise ValueError("Expected a strong positive ANC / institutional-delivery correlation")
    if correlations["ANC / high-risk cases"] > -0.70:
        raise ValueError("Expected a strong negative ANC / high-risk correlation")
    if not 0.10 < abs(correlations["ANC / immunization"]) < 0.70:
        raise ValueError("Expected a weak or moderate ANC / immunization correlation")

    return changes, {**outlier_counts, **correlations}


def print_statistics(rows: list[dict[str, object]]) -> None:
    if pd is not None:
        frame = pd.DataFrame(rows, columns=COLUMNS)
        print("\nDescriptive statistics:")
        print(frame.loc[:, NUMERIC_COLUMNS].describe().round(2).to_string())
        print("\nSample rows:")
        print(frame.head(10).to_string(index=False))
        return

    print("\nDescriptive statistics (Pandas unavailable; standard-library fallback):")
    print(f"{'':20} {'count':>7} {'mean':>9} {'std':>9} {'min':>9} {'25%':>9} {'50%':>9} {'75%':>9} {'max':>9}")
    for column in NUMERIC_COLUMNS:
        values = [float(row[column]) for row in rows]
        standard_deviation = stdev(values) if len(values) > 1 else 0.0
        summary = (len(values), mean(values), standard_deviation, min(values), quantile(values, 0.25), quantile(values, 0.5), quantile(values, 0.75), max(values))
        print(f"{column:20} {summary[0]:7.0f} {summary[1]:9.2f} {summary[2]:9.2f} {summary[3]:9.2f} {summary[4]:9.2f} {summary[5]:9.2f} {summary[6]:9.2f} {summary[7]:9.2f}")
    print("\nSample rows:")
    for row in rows[:10]:
        print(",".join(str(row[column]) for column in COLUMNS))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "data" / "healthcare.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    changes, metrics = validate(rows)

    if pd is not None:
        pd.DataFrame(rows, columns=COLUMNS).to_csv(output_path, index=False)
    else:
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved synthetic dataset: {output_path}")
    print(f"Rows: {len(rows)} | Columns: {len(COLUMNS)} | Districts: 20 | Months: 12")
    print("Validation: passed (schema, ranges, missing values, district-month uniqueness, sorting, IQR, trends, correlations)")
    print(f"Month-to-month changes >= 10%: {changes}")
    print("IQR outliers by numeric column: " + ", ".join(f"{key}={metrics[key]}" for key in NUMERIC_COLUMNS))
    print("Selected Pearson correlations: " + ", ".join(f"{key}={metrics[key]:.3f}" for key in list(metrics)[-3:]))
    print_statistics(rows)


if __name__ == "__main__":
    main()
