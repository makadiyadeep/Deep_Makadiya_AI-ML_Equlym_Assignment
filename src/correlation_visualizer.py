"""Save a Pearson correlation heatmap and print its dynamic interpretation."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.correlation_analyzer import (
    INDICATORS,
    INDICATOR_LABELS,
    analyze_correlations,
    correlation_strength,
)
from src.validator import DEFAULT_DATA_PATH, validate_dataset

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def save_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a labeled heatmap, displaying undefined coefficients as N/A."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = correlation_matrix.reindex(index=INDICATORS, columns=INDICATORS)
    values = matrix.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad(color="#d9d9d9")

    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(masked_values, cmap=colormap, vmin=-1, vmax=1)
    labels = [INDICATOR_LABELS[indicator] for indicator in INDICATORS]
    axis.set_xticks(np.arange(len(INDICATORS)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(INDICATORS)), labels)
    axis.set_xlabel("Indicator")
    axis.set_ylabel("Indicator")
    axis.set_title("Pearson Correlation Heatmap: Healthcare Indicators")

    for row in range(len(INDICATORS)):
        for column in range(len(INDICATORS)):
            value = values[row, column]
            annotation = "N/A" if not np.isfinite(value) else f"{value:.2f}"
            text_color = "white" if np.isfinite(value) and abs(value) >= 0.55 else "black"
            axis.text(column, row, annotation, ha="center", va="center", color=text_color)

    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Pearson correlation coefficient (r)")
    figure.tight_layout()
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def _valid_matrix_pairs(correlation_matrix: pd.DataFrame) -> list[tuple[str, str, float]]:
    pairs = []
    for first_index, indicator_1 in enumerate(INDICATORS):
        for indicator_2 in INDICATORS[first_index + 1 :]:
            value = correlation_matrix.at[indicator_1, indicator_2]
            if pd.notna(value) and np.isfinite(value):
                pairs.append((indicator_1, indicator_2, float(value)))
    return pairs


def _report_strongest_pair(
    pairs: list[tuple[str, str, float]],
    relationship: str,
) -> None:
    candidates = [
        pair for pair in pairs
        if (pair[2] > 0 if relationship == "positive" else pair[2] < 0)
    ]
    if not candidates:
        print(f"No defined {relationship} pairwise correlations are available.")
        return

    strongest_value = (
        max(pair[2] for pair in candidates)
        if relationship == "positive"
        else min(pair[2] for pair in candidates)
    )
    strongest_pairs = [pair for pair in candidates if pair[2] == strongest_value]
    for indicator_1, indicator_2, coefficient in strongest_pairs:
        strength = correlation_strength(coefficient)
        print(
            f"Strongest {relationship}: {INDICATOR_LABELS[indicator_1]} and "
            f"{INDICATOR_LABELS[indicator_2]} show a {strength} {relationship} "
            f"linear association (Pearson r = {coefficient:.3f})."
        )


def print_correlation_analysis(
    correlation_matrix: pd.DataFrame,
    relationships: pd.DataFrame,
    sample_size: int,
    threshold: float,
) -> None:
    """Print the strongest coefficients and every threshold-qualified pair."""
    print("\n--- Correlation Analysis ---")
    if sample_size < 30:
        print(
            f"Warning: sample size is {sample_size}; pairwise correlations may "
            "be unstable with fewer than 30 observations."
        )
    print(
        "Strength categories: strong (|r| >= 0.70), moderate "
        "(0.30 <= |r| < 0.70), weak (|r| < 0.30). These describe magnitude, "
        "not statistical significance."
    )

    pairs = _valid_matrix_pairs(correlation_matrix)
    _report_strongest_pair(pairs, "positive")
    _report_strongest_pair(pairs, "negative")

    print(f"\nPairs meeting |r| >= {threshold:.2f}:")
    if relationships.empty:
        print("No indicator pairs met the configured threshold.")
    else:
        for row in relationships.itertuples(index=False):
            first = INDICATOR_LABELS[row.indicator_1]
            second = INDICATOR_LABELS[row.indicator_2]
            print(
                f"- {first} and {second} show a {row.strength} "
                f"{row.relationship} linear association (Pearson r = "
                f"{row.correlation:.3f}) in the observed dataset."
            )

    print("\nCorrelation describes association, not causation.")
    print(
        "This dataset is synthetic; its correlations demonstrate the software's "
        "functionality, not real-world healthcare relationships."
    )


def run_workflow(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    threshold: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Validate data, calculate correlations, and save matrix and heatmap."""
    data = validate_dataset(data_path)
    correlation_matrix, relationships = analyze_correlations(data, threshold=threshold)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matrix_path = output_path / "correlation_matrix.csv"
    correlation_matrix.to_csv(matrix_path, index_label="indicator")
    heatmap_path = save_heatmap(
        correlation_matrix,
        output_path / "graphs" / "correlation_heatmap.png",
    )

    print("\nPearson correlation matrix:")
    print(correlation_matrix.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nThreshold-qualified relationships:")
    if relationships.empty:
        print("None")
    else:
        print(relationships.to_string(index=False))
    print_correlation_analysis(
        correlation_matrix,
        relationships,
        sample_size=len(data),
        threshold=threshold,
    )
    print(f"\nCorrelation matrix saved: {matrix_path}")
    print(f"Heatmap saved: {heatmap_path}")
    return correlation_matrix, relationships, matrix_path, heatmap_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Healthcare CSV path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for matrix and graph")
    parser.add_argument("--threshold", type=float, default=0.70, help="Minimum absolute Pearson coefficient")
    arguments = parser.parse_args()
    run_workflow(arguments.data, arguments.output_dir, arguments.threshold)


if __name__ == "__main__":
    main()
