"""Plot IQR outliers and print data-derived summaries."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.outlier_detector import OUTLIER_INDICATORS, detect_outliers
from src.validator import DEFAULT_DATA_PATH, validate_dataset

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "graphs"
INDICATOR_LABELS = {
    "anc_coverage": "ANC coverage",
    "institutional_delivery": "Institutional delivery",
    "immunization": "Immunization",
    "high_risk_cases": "High-risk cases",
}


def _unit(indicator: str) -> str:
    return "cases" if indicator == "high_risk_cases" else "%"


def _format_value(value: float, indicator: str) -> str:
    if indicator == "high_risk_cases":
        return f"{value:,.2f} cases"
    return f"{value:.2f}%"


def _format_month(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.strftime("%Y-%m") if not pd.isna(parsed) else str(value)


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def analyze_outliers(outliers: pd.DataFrame) -> None:
    """Print counts and the exact observations outside their IQR bounds."""
    total = len(outliers)
    low_count = int(outliers["outlier_direction"].eq("low").sum())
    high_count = int(outliers["outlier_direction"].eq("high").sum())
    print("\n--- IQR Outlier Analysis ---")
    print(f"Total outliers detected: {total}")
    print(f"Low outliers: {low_count}; high outliers: {high_count}")

    if outliers.empty:
        print("No outliers were detected by the IQR method.")
        print("Indicator with the most outliers: none")
        return

    counts = outliers["indicator"].value_counts().reindex(OUTLIER_INDICATORS, fill_value=0)
    maximum = int(counts.max())
    most_common = [INDICATOR_LABELS[indicator] for indicator in counts.index[counts.eq(maximum)]]
    print(f"Indicator(s) with the most outliers ({maximum}): {', '.join(most_common)}")

    print("Outlier observations and distance beyond the applicable bound:")
    ordered = outliers.sort_values(["indicator", "district", "month"], kind="stable")
    for row in ordered.itertuples(index=False):
        if row.outlier_direction == "low":
            distance = row.lower_bound - row.value
            bound_label = "below lower bound"
        else:
            distance = row.value - row.upper_bound
            bound_label = "above upper bound"
        unit = "cases" if row.indicator == "high_risk_cases" else "percentage points"
        print(
            f"- {row.district}, {_format_month(row.month)}, {INDICATOR_LABELS[row.indicator]}: "
            f"{_format_value(row.value, row.indicator)} ({row.outlier_direction}); "
            f"Q1={row.q1:.2f}, Q3={row.q3:.2f}, IQR={row.iqr:.2f}, "
            f"bounds=[{row.lower_bound:.2f}, {row.upper_bound:.2f}]; "
            f"{distance:.2f} {unit} {bound_label}."
        )


def _numeric_values(df: pd.DataFrame, indicator: str) -> pd.Series:
    values = pd.to_numeric(df[indicator], errors="coerce").astype(float)
    return values.loc[values.notna() & np.isfinite(values.to_numpy())]


def _plot_boxplot(df: pd.DataFrame, indicator: str, output_dir: Path) -> Path:
    values = _numeric_values(df, indicator)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    if values.empty:
        axis.text(0.5, 0.5, "No valid numeric observations", ha="center", va="center", transform=axis.transAxes)
    else:
        axis.boxplot(values, whis=1.5, showfliers=True, patch_artist=True)
    axis.set_title(f"Overall Distribution and IQR Outliers: {INDICATOR_LABELS[indicator]}")
    axis.set_ylabel(f"{INDICATOR_LABELS[indicator]} ({_unit(indicator)})")
    axis.set_xlabel("All district-month observations")
    axis.set_xticks([1], ["All observations"])
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / f"outliers_box_{indicator}.png"
    return _save_figure(figure, path)


def _plot_outlier_counts(outliers: pd.DataFrame, output_dir: Path) -> Path:
    counts = outliers["indicator"].value_counts().reindex(OUTLIER_INDICATORS, fill_value=0)
    labels = [INDICATOR_LABELS[indicator] for indicator in OUTLIER_INDICATORS]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(labels, counts.values, color="tab:blue")
    axis.bar_label(bars, padding=3)
    axis.set_title("Detected Outliers by Indicator")
    axis.set_xlabel("Indicator")
    axis.set_ylabel("Number of outlier observations")
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylim(bottom=0, top=max(1, int(counts.max()) + 1))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "outlier_counts_by_indicator.png"
    return _save_figure(figure, path)


def _plot_direction_counts(outliers: pd.DataFrame, output_dir: Path) -> Path:
    counts = (
        outliers.groupby(["indicator", "outlier_direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=OUTLIER_INDICATORS, columns=["low", "high"], fill_value=0)
    )
    positions = np.arange(len(OUTLIER_INDICATORS))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    low_bars = axis.bar(positions - width / 2, counts["low"], width, label="Low", color="tab:blue")
    high_bars = axis.bar(positions + width / 2, counts["high"], width, label="High", color="tab:orange")
    axis.bar_label(low_bars, padding=2)
    axis.bar_label(high_bars, padding=2)
    axis.set_xticks(positions, [INDICATOR_LABELS[indicator] for indicator in OUTLIER_INDICATORS], rotation=20, ha="right")
    axis.set_title("Low and High Outlier Counts by Indicator")
    axis.set_xlabel("Indicator")
    axis.set_ylabel("Number of outlier observations")
    axis.set_ylim(bottom=0, top=max(1, int(counts.to_numpy().max()) + 1))
    axis.legend(title="Outlier direction")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "outlier_counts_by_direction.png"
    return _save_figure(figure, path)


def generate_outlier_graphs(
    df: pd.DataFrame,
    outliers: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Create per-indicator box plots and two outlier-count summaries."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = [_plot_boxplot(df, indicator, output_path) for indicator in OUTLIER_INDICATORS]
    paths.append(_plot_outlier_counts(outliers, output_path))
    paths.append(_plot_direction_counts(outliers, output_path))
    for path in paths:
        print(f"Graph saved: {path}")
    return paths


def run_workflow(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    method: str = "iqr",
) -> tuple[pd.DataFrame, list[Path]]:
    """Validate the CSV, detect outliers, report them, and save charts."""
    data = validate_dataset(data_path)
    outliers = detect_outliers(data, method=method)
    analyze_outliers(outliers)
    graph_paths = generate_outlier_graphs(data, outliers, output_dir)
    print(f"Generated {len(graph_paths)} outlier graphs in {Path(output_dir)}")
    return outliers, graph_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Healthcare CSV path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PNG graphs")
    parser.add_argument("--method", choices=["iqr"], default="iqr", help="Outlier detection method")
    arguments = parser.parse_args()
    run_workflow(arguments.data, arguments.output_dir, arguments.method)


if __name__ == "__main__":
    main()
