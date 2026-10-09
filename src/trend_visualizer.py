"""Create trend charts and data-derived text analysis for healthcare data."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.trend_detector import INDICATORS, detect_trends
from src.validator import DEFAULT_DATA_PATH, validate_dataset

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "graphs"
COVERAGE_INDICATORS = (
    "anc_coverage",
    "institutional_delivery",
    "immunization",
)
INDICATOR_LABELS = {
    "anc_coverage": "ANC coverage",
    "institutional_delivery": "Institutional delivery",
    "immunization": "Immunization",
    "high_risk_cases": "High-risk cases",
}


def _is_count(indicator: str) -> bool:
    return indicator == "high_risk_cases"


def _unit(indicator: str) -> str:
    return "cases" if _is_count(indicator) else "%"


def _month_label(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def _value_label(value: float, indicator: str) -> str:
    if _is_count(indicator):
        return f"{value:,.0f} cases"
    return f"{value:.2f}%"


def _change_label(change: float, indicator: str) -> str:
    if _is_count(indicator):
        return f"{change:+,.0f} cases"
    return f"{change:+.2f} percentage points"


def _comparison_value_label(value: float, indicator: str) -> str:
    if _is_count(indicator):
        return f"{value:,.2f} cases"
    return f"{value:.2f}%"


def _relative_change(start: float, end: float) -> float | None:
    if start == 0 or not np.isfinite(start) or not np.isfinite(end):
        return None
    return ((end - start) / start) * 100


def _valid_observations(
    df: pd.DataFrame,
    indicator: str,
    district: str | None = None,
) -> pd.DataFrame:
    columns = ["month", "district", indicator]
    observations = df.loc[:, columns].copy()
    observations["month"] = pd.to_datetime(observations["month"], errors="coerce")
    observations[indicator] = pd.to_numeric(observations[indicator], errors="coerce")
    valid = (
        observations["month"].notna()
        & observations["district"].notna()
        & observations[indicator].notna()
        & np.isfinite(observations[indicator])
    )
    observations = observations.loc[valid]
    if district is not None:
        observations = observations.loc[observations["district"] == district]
    return observations.sort_values(["month", "district"], kind="stable")


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def _print_line_analysis(
    df: pd.DataFrame,
    indicator: str,
    district: str,
    threshold: float,
) -> None:
    observations = _valid_observations(df, indicator, district)
    label = INDICATOR_LABELS[indicator]
    print(f"\nAnalysis for monthly chart: {label} ({district})")
    if observations.empty:
        print("Observed fact: no valid observations are available for this district and indicator.")
        return

    values = observations[indicator]
    first = observations.iloc[0]
    last = observations.iloc[-1]
    start_value = float(first[indicator])
    end_value = float(last[indicator])
    absolute_change = end_value - start_value
    percentage_change = _relative_change(start_value, end_value)

    if len(observations) < 2:
        direction = "undetermined (fewer than two observations)"
    elif percentage_change is None:
        direction = (
            "increasing" if absolute_change > 0
            else "decreasing" if absolute_change < 0
            else "relatively stable"
        ) + " (relative threshold comparison unavailable because the starting value is zero)"
    elif abs(percentage_change) < threshold or absolute_change == 0:
        direction = "relatively stable"
    else:
        direction = "increasing" if absolute_change > 0 else "decreasing"

    print(f"Interpretation: overall direction is {direction} (based on the {threshold:g}% endpoint relative-change threshold).")
    print(
        f"Observed fact: {first['month']:%Y-%m} started at {_value_label(start_value, indicator)}; "
        f"{last['month']:%Y-%m} ended at {_value_label(end_value, indicator)}."
    )
    change_description = _change_label(absolute_change, indicator)
    if percentage_change is None:
        relative_description = "relative percentage change unavailable (zero starting value)"
    else:
        relative_description = f"relative percentage change {percentage_change:+.2f}%"
    print(f"Observed fact: net change was {change_description}; {relative_description}.")

    maximum = values.max()
    minimum = values.min()
    highest_months = [_month_label(value) for value in observations.loc[values.eq(maximum), "month"]]
    lowest_months = [_month_label(value) for value in observations.loc[values.eq(minimum), "month"]]
    print(
        f"Observed fact: highest value {_value_label(float(maximum), indicator)} in "
        f"{', '.join(highest_months)}; lowest value {_value_label(float(minimum), indicator)} "
        f"in {', '.join(lowest_months)}."
    )


def _plot_monthly_indicator(
    df: pd.DataFrame,
    indicator: str,
    selected_district: str,
    output_dir: Path,
    threshold: float,
) -> Path:
    observations = _valid_observations(df, indicator)
    monthly = (
        observations.groupby("month")[indicator]
        .agg(mean="mean", minimum="min", maximum="max")
        .reset_index()
    )
    selected = _valid_observations(df, indicator, selected_district)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    if not monthly.empty:
        axis.fill_between(
            monthly["month"],
            monthly["minimum"],
            monthly["maximum"],
            alpha=0.15,
            color="tab:blue",
            label="District range",
        )
        axis.plot(
            monthly["month"],
            monthly["mean"],
            marker="o",
            color="tab:blue",
            label="Monthly district mean",
        )
    if not selected.empty:
        axis.plot(
            selected["month"],
            selected[indicator],
            marker="o",
            linewidth=2,
            color="tab:orange",
            label=f"{selected_district}",
        )
    if monthly.empty:
        axis.text(0.5, 0.5, "No valid observations", ha="center", va="center", transform=axis.transAxes)

    axis.set_title(f"Monthly {INDICATOR_LABELS[indicator]} Trends")
    axis.set_xlabel("Month")
    axis.set_ylabel(f"{INDICATOR_LABELS[indicator]} ({_unit(indicator)})")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    if not monthly.empty or not selected.empty:
        axis.legend()
    figure.tight_layout()

    path = _save_figure(figure, output_dir / f"monthly_{indicator}.png")
    _print_line_analysis(df, indicator, selected_district, threshold)
    print(f"Graph saved: {path}")
    return path


def _tied_districts(values: pd.Series, target: float) -> list[str]:
    return [str(district) for district in values.index[values.eq(target)]]


def _plot_district_comparison(
    df: pd.DataFrame,
    indicator: str,
    output_dir: Path,
    period_label: str,
) -> Path:
    observations = _valid_observations(df, indicator)
    averages = observations.groupby("district")[indicator].mean().sort_values()
    figure, axis = plt.subplots(figsize=(11, max(6, len(averages) * 0.34)))

    if averages.empty:
        axis.text(0.5, 0.5, "No valid observations", ha="center", va="center", transform=axis.transAxes)
        print(f"\nAnalysis for district comparison: {INDICATOR_LABELS[indicator]}")
        print("Observed fact: no valid district averages are available.")
    else:
        axis.barh(averages.index, averages.values, color="tab:green")
        axis.set_xlabel(f"Mean {INDICATOR_LABELS[indicator]} ({_unit(indicator)})")
        axis.set_ylabel("District")
        axis.grid(axis="x", alpha=0.25)
        print(f"\nAnalysis for district comparison: {INDICATOR_LABELS[indicator]} ({period_label})")
        if len(averages) < 2:
            print("Observed fact: fewer than two districts have valid observations; no between-district difference is available.")
        else:
            low = float(averages.iloc[0])
            high = float(averages.iloc[-1])
            lowest = _tied_districts(averages, low)
            highest = _tied_districts(averages, high)
            difference = high - low
            difference_unit = "cases" if _is_count(indicator) else "percentage points"
            print(
                f"Observed fact: highest period mean {_comparison_value_label(high, indicator)} "
                f"({', '.join(highest)}); lowest period mean {_comparison_value_label(low, indicator)} "
                f"({', '.join(lowest)}); difference {difference:.2f} {difference_unit}."
            )

    axis.set_title(f"District Comparison: {INDICATOR_LABELS[indicator]} ({period_label})")
    figure.tight_layout()
    path = _save_figure(figure, output_dir / f"district_comparison_{indicator}.png")
    print(f"Graph saved: {path}")
    return path


def _describe_event(row: pd.Series, indicator: str) -> str:
    month = _month_label(row["month"])
    change = float(row["absolute_change"])
    if _is_count(indicator):
        amount = f"{abs(change):,.0f} cases"
    else:
        amount = f"{abs(change):.2f} percentage points"
    return (
        f"{row['district']} in {month}: {row['direction']} by {amount} "
        f"({float(row['percentage_change']):+.2f}% relative change)"
    )


def _plot_percentage_changes(
    trends: pd.DataFrame,
    indicator: str,
    output_dir: Path,
    threshold: float,
) -> Path:
    changes = trends.loc[trends["indicator"] == indicator].sort_values(
        ["month", "district"], kind="stable"
    )
    figure, axis = plt.subplots(figsize=(max(10, len(changes) * 0.42), 5.5))

    print(f"\nAnalysis for detected percentage changes: {INDICATOR_LABELS[indicator]}")
    if changes.empty:
        axis.text(0.5, 0.5, "No significant trends detected", ha="center", va="center", transform=axis.transAxes)
        print(f"Observed fact: no trends met the {threshold:g}% threshold for this indicator.")
    else:
        positions = np.arange(len(changes))
        colors = changes["direction"].map({"increase": "tab:green", "decrease": "tab:red"})
        axis.bar(positions, changes["percentage_change"].astype(float), color=colors)
        labels = [f"{row.district}\n{_month_label(row.month)}" for row in changes.itertuples()]
        axis.set_xticks(positions, labels, rotation=60, ha="right")
        axis.axhline(0, color="black", linewidth=0.8)

        positive = changes.loc[changes["percentage_change"] > 0]
        negative = changes.loc[changes["percentage_change"] < 0]
        if positive.empty:
            print("Observed fact: no positive significant percentage changes were detected.")
        else:
            largest_positive = float(positive["percentage_change"].max())
            rows = positive.loc[positive["percentage_change"].eq(largest_positive)]
            for _, row in rows.iterrows():
                print("Observed fact: largest positive detected change: " + _describe_event(row, indicator) + ".")
        if negative.empty:
            print("Observed fact: no negative significant percentage changes were detected.")
        else:
            largest_negative = float(negative["percentage_change"].min())
            rows = negative.loc[negative["percentage_change"].eq(largest_negative)]
            for _, row in rows.iterrows():
                print("Observed fact: largest negative detected change: " + _describe_event(row, indicator) + ".")
        if _is_count(indicator):
            increases = int(changes["direction"].eq("increase").sum())
            decreases = int(changes["direction"].eq("decrease").sum())
            print(f"Observed fact: detected case-count changes include {increases} increases and {decreases} decreases.")

    axis.set_title(f"Detected Month-over-Month Percentage Changes: {INDICATOR_LABELS[indicator]}")
    axis.set_xlabel("District and month")
    axis.set_ylabel("Relative change (%)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = _save_figure(figure, output_dir / f"percentage_changes_{indicator}.png")
    print(f"Graph saved: {path}")
    return path


def _plot_direction_summary(trends: pd.DataFrame, output_dir: Path) -> Path:
    counts = trends["direction"].value_counts().reindex(["increase", "decrease"], fill_value=0)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(["Increase", "Decrease"], counts.values, color=["tab:green", "tab:red"])
    axis.set_title("Detected Trend Directions")
    axis.set_xlabel("Direction")
    axis.set_ylabel("Number of detected trends")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    total = int(counts.sum())
    increases = int(counts["increase"])
    decreases = int(counts["decrease"])
    print("\nAnalysis for trend direction summary")
    print(f"Observed fact: {increases} increasing trends and {decreases} decreasing trends; total {total}.")
    if total == 0:
        print("Observed fact: no significant trends were detected.")
    elif increases > decreases:
        print("Observed fact: increases are more common among detected trends.")
    elif decreases > increases:
        print("Observed fact: decreases are more common among detected trends.")
    else:
        print("Observed fact: increases and decreases are tied.")

    path = _save_figure(figure, output_dir / "trend_direction_summary.png")
    print(f"Graph saved: {path}")
    return path


def _plot_selected_district(
    df: pd.DataFrame,
    district: str,
    output_dir: Path,
) -> Path:
    district_data = df.loc[df["district"] == district].copy()
    district_data["month"] = pd.to_datetime(district_data["month"], errors="coerce")
    district_data = district_data.dropna(subset=["month"]).sort_values("month")

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for indicator in COVERAGE_INDICATORS:
        observations = _valid_observations(df, indicator, district)
        axes[0].plot(
            observations["month"],
            observations[indicator],
            marker="o",
            label=INDICATOR_LABELS[indicator],
        )
    axes[0].set_title(f"Coverage Indicators: {district}")
    axes[0].set_ylabel("Coverage (%)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    risk = _valid_observations(df, "high_risk_cases", district)
    if risk.empty:
        axes[1].text(0.5, 0.5, "No valid high-risk observations", ha="center", va="center", transform=axes[1].transAxes)
    else:
        axes[1].plot(risk["month"], risk["high_risk_cases"], marker="o", color="tab:red", label="High-risk cases")
        axes[1].legend()
    axes[1].set_title(f"High-risk Case Counts: {district}")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Cases")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()

    valid_months = district_data["month"].drop_duplicates()
    print(f"\nAnalysis for district-level comparison: {district}")
    if valid_months.empty:
        print("Observed fact: no valid monthly observations are available for this district.")
    else:
        print(
            f"Observed fact: chart covers {len(valid_months)} observed months, "
            f"from {_month_label(valid_months.min())} through {_month_label(valid_months.max())}. "
            "Coverage percentages and case counts are shown in separate panels."
        )

    path = _save_figure(figure, output_dir / f"district_trends_{district}.png")
    print(f"Graph saved: {path}")
    return path


def generate_visualizations(
    df: pd.DataFrame,
    trends: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    district: str | None = None,
    threshold: float = 10.0,
) -> list[Path]:
    """Save charts and print data-derived analysis for each generated chart."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    districts = sorted(df["district"].dropna().unique().tolist())
    if not districts:
        raise ValueError("Cannot visualize data without a valid district")
    selected_district = district or districts[0]
    if selected_district not in districts:
        raise ValueError(f"Unknown district {selected_district!r}; available districts: {districts}")

    valid_months = pd.to_datetime(df["month"], errors="coerce").dropna()
    if valid_months.empty:
        period_label = "no valid months"
    else:
        period_label = f"{valid_months.min():%Y-%m} to {valid_months.max():%Y-%m}"

    print(f"\nSelected district for district-specific charts: {selected_district}")
    print(f"Observed data period: {period_label}")
    graph_paths: list[Path] = []

    for indicator in INDICATORS:
        graph_paths.append(
            _plot_monthly_indicator(df, indicator, selected_district, output_path, threshold)
        )
    for indicator in INDICATORS:
        graph_paths.append(
            _plot_district_comparison(df, indicator, output_path, period_label)
        )
    for indicator in INDICATORS:
        graph_paths.append(
            _plot_percentage_changes(trends, indicator, output_path, threshold)
        )
    graph_paths.append(_plot_direction_summary(trends, output_path))
    graph_paths.append(_plot_selected_district(df, selected_district, output_path))

    print(f"\nGenerated {len(graph_paths)} graph files in {output_path}")
    return graph_paths


def run_workflow(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    district: str | None = None,
    threshold: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Validate data, detect trends, save charts, and print analysis."""
    data = validate_dataset(data_path)
    trends = detect_trends(data, threshold=threshold)
    increases = int(trends["direction"].eq("increase").sum())
    decreases = int(trends["direction"].eq("decrease").sum())
    print(
        f"\nDetected {len(trends)} significant trends at a {threshold:g}% threshold "
        f"({increases} increases, {decreases} decreases)."
    )
    if trends.empty:
        print("No significant trends were detected; percentage-change charts will be empty.")
    graph_paths = generate_visualizations(
        data,
        trends,
        output_dir=output_dir,
        district=district,
        threshold=threshold,
    )
    return data, trends, graph_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Healthcare CSV path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PNG graphs")
    parser.add_argument("--district", help="District to highlight (defaults to the first district alphabetically)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Minimum absolute percentage change")
    arguments = parser.parse_args()
    run_workflow(
        data_path=arguments.data,
        output_dir=arguments.output_dir,
        district=arguments.district,
        threshold=arguments.threshold,
    )


if __name__ == "__main__":
    main()
