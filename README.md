# Automated Insight Generation Engine

An educational Python project that validates district-month healthcare data, detects trends and statistical outliers, measures Pearson associations, and turns calculated results into structured insights with configurable engineering severity labels. A Streamlit dashboard presents the analyses and downloadable reports.

> **Synthetic data:** `data/healthcare.csv` is generated for software testing and education. It is not sourced from health departments and does not represent actual healthcare statistics for Gujarat. Results from it demonstrate application behavior, not real-world health relationships.

## Workflow

The workflow validates the input CSV, calculates analytical results, then presents them in the dashboard or saves reports through the analysis modules:

1. `src/validator.py` checks required columns, missing values, duplicate district-month records, dates, numeric conversion, coverage ranges, and high-risk counts. It returns a converted in-memory DataFrame and does not rewrite the source CSV.
2. `src/trend_detector.py` compares adjacent months within each district. The default detection threshold is an absolute month-over-month relative change of 10%.
3. `src/outlier_detector.py` applies the IQR rule independently to each indicator across the complete dataset: values below $Q1 - 1.5 \times IQR$ or above $Q3 + 1.5 \times IQR$ are flagged.
4. `src/correlation_analyzer.py` calculates pairwise Pearson coefficients using pairwise-complete numeric observations. The default reporting threshold is $|r| \geq 0.70$.
5. `src/insight_generator.py` combines those existing results into a consistent insight schema and assigns severity through `src/severity.py`.
6. `app.py` runs the analyses and provides interactive views and downloads.

The analysis modules can also be run individually from the project root:

```sh
.venv/bin/python -m src.trend_visualizer
.venv/bin/python -m src.outlier_visualizer
.venv/bin/python -m src.correlation_visualizer
.venv/bin/python -m src.insight_generator
```

## Dataset

The default input is `data/healthcare.csv`. It has one observation per district per month; the included synthetic file contains 240 records for 20 districts across January–December 2025.

The expected CSV columns and order are:

| Column | Expected value |
|---|---|
| `month` | Date formatted `YYYY-MM-01` |
| `district` | District name as text |
| `anc_coverage` | Numeric ANC coverage percentage, 0–100 |
| `institutional_delivery` | Numeric institutional-delivery percentage, 0–100 |
| `immunization` | Numeric immunization percentage, 0–100 |
| `high_risk_cases` | Non-negative integer count |

The dashboard accepts either this project CSV or an uploaded CSV with the required schema. Uploaded files are validated before analysis; invalid files are not silently repaired or written over the original data.

## Setup

From the project root on macOS/Linux, create and populate the virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Launch the dashboard:

```sh
.venv/bin/python -m streamlit run app.py
```

Run the test suite:

```sh
.venv/bin/python -m pytest -q
```

To regenerate the deterministic synthetic dataset:

```sh
.venv/bin/python scripts/generate_dataset.py
```

Dependencies are listed in `requirements.txt` (including Pandas, NumPy, Matplotlib, Plotly, Streamlit, and pytest).

## Dashboard

The dashboard includes dataset overview, automated insights, trend analysis, outlier analysis, correlation analysis, data visualizations, and downloads.

Sidebar controls filter displayed district, month, healthcare indicators, insight type, and severity. The Pearson reporting threshold is adjustable, initially 0.70. District, month, indicator, type, and severity filters affect displayed results and relevant charts; they do not change the observation population used to calculate correlations or IQR bounds. Correlation displays use the full validated dataset, and outlier box plots show the full-dataset distribution. The UI labels monthly high-risk summaries as counts (or means across selected districts), not percentages.

Charts include monthly indicator lines, district comparisons, detected month-over-month changes, IQR box plots, outlier counts by indicator/direction, and a Pearson correlation heatmap. The dashboard offers filtered insights CSV, all insights CSV, correlation matrix CSV, and structured insights JSON downloads.

Standalone visualization modules save their generated files in `outputs/graphs/`; the dashboard renders its charts in the page.

## Severity Rules

The default severity cutoffs are configurable engineering choices, not medically validated thresholds:

- Trends: Low below 15% absolute change; Medium from 15% to below 30%; High at or above 30%.
- Outliers: severity is based on distance beyond the applicable IQR fence: Low up to 0.5 IQR, Medium above 0.5 through 1.5 IQR, and High above 1.5 IQR.
- Correlations: Low severity; a large coefficient alone is not an emergency or evidence of harm.

These labels are for software prioritization only and must not be treated as clinical advice.

## Outputs

Generated files currently present in `outputs/` include:

```text
outputs/
├── correlation_matrix.csv
├── insights.csv
├── sample_insights.json
└── graphs/
	├── correlation_heatmap.png
	├── district_comparison_anc_coverage.png
	├── district_comparison_high_risk_cases.png
	├── district_comparison_immunization.png
	├── district_comparison_institutional_delivery.png
	├── district_trends_Ahmedabad.png
	├── monthly_anc_coverage.png
	├── monthly_high_risk_cases.png
	├── monthly_immunization.png
	├── monthly_institutional_delivery.png
	├── outlier_counts_by_direction.png
	├── outlier_counts_by_indicator.png
	├── outliers_box_anc_coverage.png
	├── outliers_box_high_risk_cases.png
	├── outliers_box_immunization.png
	├── outliers_box_institutional_delivery.png
	├── percentage_changes_anc_coverage.png
	├── percentage_changes_high_risk_cases.png
	├── percentage_changes_immunization.png
	├── percentage_changes_institutional_delivery.png
	└── trend_direction_summary.png
```

The district-specific trend chart filename uses the selected district name; the listing reflects the current default-dataset output.

## Limitations

- The included data is synthetic, so detected patterns and correlations are not evidence about actual healthcare outcomes.
- Pearson correlation measures linear association, not causation, and does not calculate statistical significance. It can be sensitive to outliers and unusual distributions.
- Correlations with fewer than three pairwise-complete observations are undefined. The analyzer warns when the total or pairwise sample is below 30; small-sample estimates may be unstable.
- Trend detection is a month-to-month threshold rule; it is not a statistical trend test and does not adjust for seasonality.
- IQR flags indicate values unusual relative to the included distribution. They do not by themselves imply a healthcare emergency.
- Severity cutoffs are configurable engineering defaults, not clinically validated risk levels.
