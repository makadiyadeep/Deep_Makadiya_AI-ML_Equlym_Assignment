from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st

from src.correlation_analyzer import INDICATORS, INDICATOR_LABELS, analyze_correlations
from src.correlation_visualizer import save_heatmap
from src.insight_generator import INSIGHT_COLUMNS, generate_insights
from src.outlier_detector import detect_outliers
from src.outlier_visualizer import generate_outlier_graphs
from src.trend_detector import detect_trends
from src.validator import DEFAULT_DATA_PATH, validate_dataset


INDICATOR_OPTIONS = {INDICATOR_LABELS[name]: name for name in INDICATORS}
DISPLAY_INDICATORS = tuple(INDICATORS)
DISTRICT_ALL = "All districts"
INSIGHT_TYPES = ("trend", "outlier", "correlation")
SEVERITIES = ("Low", "Medium", "High")


@st.cache_data(show_spinner=False)
def _load_validated_csv(csv_content: bytes) -> tuple[pd.DataFrame, str]:
    """Run the existing path-based validator against a temporary upload copy."""
    temporary_path: Path | None = None
    report = StringIO()
    try:
        with NamedTemporaryFile(suffix=".csv", delete=False) as temporary_file:
            temporary_file.write(csv_content)
            temporary_path = Path(temporary_file.name)
        with redirect_stdout(report):
            validated = validate_dataset(temporary_path)
        return validated, report.getvalue()
    except Exception as error:
        details = report.getvalue().strip()
        raise ValueError(f"{details}\n{error}".strip()) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_trend_and_outlier_results(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return detect_trends(data), detect_outliers(data)


@st.cache_data(show_spinner=False)
def _cached_correlations(
    data: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        matrix, relationships = analyze_correlations(data, threshold=threshold)
    return matrix, relationships, [str(item.message) for item in captured]


@st.cache_data(show_spinner=False)
def _cached_insights(
    data: pd.DataFrame,
    trends: pd.DataFrame,
    outliers: pd.DataFrame,
    matrix: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    return generate_insights(data, trends, outliers, (matrix, relationships))


@st.cache_data(show_spinner=False)
def _heatmap_image(matrix: pd.DataFrame, indicators: tuple[str, ...]) -> bytes:
    with TemporaryDirectory() as temporary_directory:
        image_path = save_heatmap(
            matrix,
            Path(temporary_directory) / "correlation_heatmap.png",
            indicators=indicators,
        )
        return image_path.read_bytes()


@st.cache_data(show_spinner=False)
def _outlier_boxplot_images(
    data: pd.DataFrame,
    outliers: pd.DataFrame,
) -> dict[str, bytes]:
    with TemporaryDirectory() as temporary_directory:
        with redirect_stdout(StringIO()):
            paths = generate_outlier_graphs(data, outliers, temporary_directory)
        return {
            path.name: path.read_bytes()
            for path in paths
            if path.name.startswith("outliers_box_")
        }


def _month_keys(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m")


def _filter_result_rows(
    frame: pd.DataFrame,
    district: str,
    months: list[str],
    indicators: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame.copy()
    if district != DISTRICT_ALL:
        filtered = filtered.loc[filtered["district"] == district]
    if months:
        filtered = filtered.loc[_month_keys(filtered["month"]).isin(months)]
    else:
        filtered = filtered.iloc[0:0]
    if indicators:
        filtered = filtered.loc[filtered["indicator"].isin(indicators)]
    else:
        filtered = filtered.iloc[0:0]
    return filtered


def _filter_insights(
    insights: pd.DataFrame,
    district: str,
    months: list[str],
    indicators: list[str],
    insight_types: list[str],
    severities: list[str],
) -> pd.DataFrame:
    if insights.empty or not insight_types or not severities or not indicators:
        return insights.iloc[0:0].copy()

    keep = []
    for row in insights.itertuples(index=False):
        if row.type not in insight_types or row.severity not in severities:
            keep.append(False)
            continue

        if row.type == "correlation":
            pair = {value.strip() for value in str(row.indicator).split(",")}
            keep.append(pair.issubset(set(indicators)))
            continue

        matches_district = district == DISTRICT_ALL or row.entity == district
        matches_month = bool(months) and row.period in months
        matches_indicator = row.indicator in indicators
        keep.append(matches_district and matches_month and matches_indicator)

    return insights.loc[keep].copy()


def _filter_results_to_insights(
    results: pd.DataFrame,
    filtered_insights: pd.DataFrame,
    insight_type: str,
) -> pd.DataFrame:
    """Apply display severity/type filters to matching detected result rows."""
    if results.empty:
        return results.copy()
    matching_insights = filtered_insights.loc[
        filtered_insights["type"] == insight_type
    ]
    allowed = {
        (str(row.entity), str(row.indicator), str(row.period))
        for row in matching_insights.itertuples(index=False)
    }
    periods = _month_keys(results["month"])
    keep = [
        (str(district), str(indicator), str(period)) in allowed
        for district, indicator, period in zip(
            results["district"],
            results["indicator"],
            periods,
        )
    ]
    return results.loc[keep].copy()


def _filtered_dataset(
    data: pd.DataFrame,
    district: str,
    months: list[str],
) -> pd.DataFrame:
    if not months:
        return data.iloc[0:0].copy()
    keys = _month_keys(data["month"])
    mask = keys.isin(months)
    if district != DISTRICT_ALL:
        mask &= data["district"].eq(district)
    return data.loc[mask].copy()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


st.set_page_config(
    page_title="Automated Insight Generation",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Automated Insight Generation Engine")
st.caption("District-level healthcare performance | Analysis is descriptive, not causal.")

with st.sidebar:
    st.header("Dataset")
    dataset_source = st.radio(
        "Choose a source",
        ("Project CSV", "Upload CSV"),
        horizontal=True,
    )

    if dataset_source == "Project CSV":
        if not DEFAULT_DATA_PATH.exists():
            st.error(f"Default dataset was not found: {DEFAULT_DATA_PATH}")
            st.stop()
        csv_content = DEFAULT_DATA_PATH.read_bytes()
        source_label = "data/healthcare.csv"
    else:
        uploaded_file = st.file_uploader("Upload a healthcare CSV", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV with the required healthcare columns to begin.")
            st.stop()
        csv_content = uploaded_file.getvalue()
        source_label = uploaded_file.name

    correlation_threshold = st.slider(
        "Absolute correlation threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.70,
        step=0.01,
        help="Changes which full-dataset Pearson pairs are reported; it does not filter observations.",
    )

try:
    with st.spinner("Validating the dataset and calculating full-dataset analyses..."):
        data, validation_report = _load_validated_csv(csv_content)
        trend_results, outlier_results = _cached_trend_and_outlier_results(data)
        correlation_matrix, correlation_pairs, correlation_warnings = _cached_correlations(
            data,
            correlation_threshold,
        )
        all_insights = _cached_insights(
            data,
            trend_results,
            outlier_results,
            correlation_matrix,
            correlation_pairs,
        )
except Exception as error:
    st.error("The selected dataset could not be validated or analyzed.")
    st.code(str(error))
    st.stop()

if dataset_source == "Project CSV":
    st.warning(
        "The project dataset is synthetic and intended for software testing. "
        "It does not represent actual healthcare statistics."
    )
else:
    st.info("Uploaded data is analyzed as provided. Correlation does not establish causation.")

with st.sidebar:
    st.success(f"Validated {len(data):,} records")
    with st.expander("Validation report"):
        st.code(validation_report or "Validation completed.")

    st.divider()
    st.header("Display filters")
    available_districts = sorted(data["district"].dropna().astype(str).unique().tolist())
    district_options = [DISTRICT_ALL, *available_districts]
    selected_district = st.selectbox("District", district_options)

    available_months = sorted(_month_keys(data["month"]).dropna().unique().tolist())
    selected_months = st.multiselect(
        "Months",
        options=available_months,
        default=available_months,
    )
    selected_indicator_labels = st.multiselect(
        "Healthcare indicators",
        options=list(INDICATOR_OPTIONS),
        default=list(INDICATOR_OPTIONS),
    )
    selected_indicators = [
        INDICATOR_OPTIONS[label] for label in selected_indicator_labels
    ]
    selected_types = st.multiselect(
        "Insight type",
        options=list(INSIGHT_TYPES),
        default=list(INSIGHT_TYPES),
    )
    selected_severities = st.multiselect(
        "Severity",
        options=list(SEVERITIES),
        default=list(SEVERITIES),
    )

filtered_data = _filtered_dataset(data, selected_district, selected_months)
filtered_trends = _filter_result_rows(
    trend_results,
    selected_district,
    selected_months,
    selected_indicators,
)
filtered_outliers = _filter_result_rows(
    outlier_results,
    selected_district,
    selected_months,
    selected_indicators,
)
filtered_insights = _filter_insights(
    all_insights,
    selected_district,
    selected_months,
    selected_indicators,
    selected_types,
    selected_severities,
)
filtered_trends = _filter_results_to_insights(
    filtered_trends,
    filtered_insights,
    "trend",
)
filtered_outliers = _filter_results_to_insights(
    filtered_outliers,
    filtered_insights,
    "outlier",
)
filtered_correlations = correlation_pairs.loc[
    correlation_pairs["indicator_1"].isin(selected_indicators)
    & correlation_pairs["indicator_2"].isin(selected_indicators)
].copy()
if "correlation" not in selected_types or "Low" not in selected_severities:
    filtered_correlations = filtered_correlations.iloc[0:0].copy()

st.caption(
    "Overview metrics and statistical results use the full validated dataset. "
    "District, month, indicator, type, and severity filters change displayed rows/charts only. "
    "Outlier bounds and correlations are not recalculated on a filtered subset."
)

overview_tab, insights_tab, trends_tab, outliers_tab, correlations_tab, visuals_tab, downloads_tab = st.tabs(
    [
        "Dataset overview",
        "Automated insights",
        "Trend analysis",
        "Outlier analysis",
        "Correlation analysis",
        "Data visualizations",
        "Downloads",
    ]
)

with overview_tab:
    st.subheader("Full-dataset overview")
    metric_rows = [
        ("Total records", f"{len(data):,}"),
        ("Districts", f"{data['district'].nunique():,}"),
        ("Months", f"{data['month'].nunique():,}"),
        ("Trends detected", f"{len(trend_results):,}"),
        ("Outliers detected", f"{len(outlier_results):,}"),
        ("Significant correlations", f"{len(correlation_pairs):,}"),
        ("Insights generated", f"{len(all_insights):,}"),
    ]
    for start in range(0, len(metric_rows), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, metric_rows[start : start + 4]):
            column.metric(label, value)

    st.caption(f"Current source: {source_label}")
    if data.empty:
        st.info("The validated dataset contains no records.")
    else:
        st.dataframe(data.head(20), width="stretch", hide_index=True)
        st.caption("Preview of the first 20 validated rows.")

with insights_tab:
    st.subheader("Generated insights")
    severity_counts = (
        filtered_insights["severity"].value_counts()
        .reindex(SEVERITIES, fill_value=0)
    )
    severity_columns = st.columns(len(SEVERITIES))
    for column, severity in zip(severity_columns, SEVERITIES):
        column.metric(f"{severity} severity", int(severity_counts[severity]))
    if filtered_insights.empty:
        st.info("No insights match the current display filters.")
    else:
        st.dataframe(
            filtered_insights.loc[:, INSIGHT_COLUMNS],
            width="stretch",
            hide_index=True,
        )
    if selected_district != DISTRICT_ALL or len(selected_months) != len(available_months):
        st.caption(
            "Correlation insights describe the full dataset and have no district/month; "
            "those filters do not remove them. Indicator, type, and severity filters do apply."
        )

with trends_tab:
    st.subheader("Significant month-over-month changes")
    trend_columns = st.columns(3)
    trend_columns[0].metric("Displayed trends", f"{len(filtered_trends):,}")
    trend_columns[1].metric(
        "Increases",
        int(filtered_trends["direction"].eq("increase").sum()),
    )
    trend_columns[2].metric(
        "Decreases",
        int(filtered_trends["direction"].eq("decrease").sum()),
    )
    if filtered_trends.empty:
        st.info("No detected trends match the current display filters.")
    else:
        st.dataframe(filtered_trends, width="stretch", hide_index=True)
        explanations = filtered_insights.loc[filtered_insights["type"] == "trend", "explanation"]
        if not explanations.empty:
            with st.expander("Generated trend explanations"):
                for explanation in explanations:
                    st.write(f"- {explanation}")

with outliers_tab:
    st.subheader("IQR outlier records")
    st.caption("IQR bounds were calculated per indicator over the complete validated dataset.")
    outlier_columns = st.columns(3)
    outlier_columns[0].metric("Displayed outliers", f"{len(filtered_outliers):,}")
    outlier_columns[1].metric(
        "Low",
        int(filtered_outliers["outlier_direction"].eq("low").sum()),
    )
    outlier_columns[2].metric(
        "High",
        int(filtered_outliers["outlier_direction"].eq("high").sum()),
    )
    if filtered_outliers.empty:
        st.info("No outlier records match the current display filters.")
    else:
        st.dataframe(filtered_outliers, width="stretch", hide_index=True)
        explanations = filtered_insights.loc[filtered_insights["type"] == "outlier", "explanation"]
        if not explanations.empty:
            with st.expander("Generated outlier explanations"):
                for explanation in explanations:
                    st.write(f"- {explanation}")

    counts = (
        filtered_outliers.groupby(["indicator", "outlier_direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=selected_indicators, columns=["low", "high"], fill_value=0)
    )
    if not counts.empty:
        counts.index = [INDICATOR_LABELS[indicator] for indicator in counts.index]
        st.bar_chart(counts, y_label="Detected outliers", color=["#3685c5", "#dc6b4a"])
    else:
        st.info("Select at least one indicator to display outlier counts.")

with correlations_tab:
    st.subheader("Pearson correlations")
    st.info("Correlation measures association, not causation. The project dataset is synthetic.")
    st.caption(
        f"Matrix calculated across all {len(data):,} validated records at the pairwise level. "
        f"The configured reporting threshold is |r| >= {correlation_threshold:.2f}; "
        "district and month display filters do not change the correlation population."
    )
    for warning in correlation_warnings:
        st.warning(warning)
    if selected_indicators:
        matrix_image = _heatmap_image(correlation_matrix, tuple(selected_indicators))
        st.image(matrix_image, caption="Full-dataset Pearson correlation heatmap")
        matrix_display = correlation_matrix.rename(
            index=INDICATOR_LABELS,
            columns=INDICATOR_LABELS,
        ).loc[
            [INDICATOR_LABELS[indicator] for indicator in selected_indicators],
            [INDICATOR_LABELS[indicator] for indicator in selected_indicators],
        ]
        st.dataframe(
            matrix_display.style.background_gradient(
                cmap="coolwarm",
                axis=None,
                vmin=-1,
                vmax=1,
            ).format("{:.3f}", na_rep="N/A"),
            width="stretch",
        )
    else:
        st.info("Select indicators in the sidebar to show the correlation heatmap and matrix.")

    st.markdown("**Pairs meeting the configured threshold**")
    if filtered_correlations.empty:
        st.info("No selected indicator pairs meet the configured threshold.")
    else:
        st.dataframe(filtered_correlations, width="stretch", hide_index=True)
        correlation_explanations = filtered_insights.loc[
            filtered_insights["type"] == "correlation", "explanation"
        ]
        for explanation in correlation_explanations:
            st.write(explanation)

with visuals_tab:
    st.subheader("Filtered trend visualizations")
    if filtered_data.empty:
        st.info("No records match the selected district and month filters.")
    else:
        coverage_indicators = [
            indicator for indicator in selected_indicators
            if indicator != "high_risk_cases"
        ]
        if coverage_indicators:
            monthly_coverage = (
                filtered_data.groupby("month", as_index=False)[coverage_indicators]
                .mean()
                .sort_values("month")
                .set_index("month")
                .rename(columns=INDICATOR_LABELS)
            )
            district_label = (
                selected_district
                if selected_district != DISTRICT_ALL
                else f"mean across {filtered_data['district'].nunique()} displayed districts"
            )
            st.markdown(f"**Coverage indicators (%) — {district_label}**")
            st.line_chart(monthly_coverage, y_label="Coverage (%)")
        if "high_risk_cases" in selected_indicators:
            monthly_counts = (
                filtered_data.groupby("month", as_index=False)["high_risk_cases"]
                .mean()
                .sort_values("month")
                .set_index("month")
            )
            count_label = (
                f"High-risk case counts — {selected_district}"
                if selected_district != DISTRICT_ALL
                else "Mean high-risk case count across displayed districts"
            )
            st.markdown(f"**{count_label} — separate count scale**")
            st.line_chart(monthly_counts, y_label="Cases")

    st.markdown("**District comparison over selected months**")
    period_data = data.loc[_month_keys(data["month"]).isin(selected_months)].copy()
    comparison_indicators = [
        indicator for indicator in selected_indicators
    ]
    if period_data.empty or not comparison_indicators:
        st.info("Select one or more months and indicators to display district comparisons.")
    else:
        district_means = period_data.groupby("district")[comparison_indicators].mean().reset_index()
        for indicator in comparison_indicators:
            district_means["selection"] = district_means["district"].map(
                lambda name: "Selected district" if name == selected_district else "Other districts"
            )
            figure = px.bar(
                district_means.sort_values(indicator),
                x="district",
                y=indicator,
                color="selection" if selected_district != DISTRICT_ALL else None,
                color_discrete_map={
                    "Selected district": "#248f8d",
                    "Other districts": "#8da6a3",
                },
                labels={
                    "district": "District",
                    indicator: f"{INDICATOR_LABELS[indicator]} ({'cases' if indicator == 'high_risk_cases' else '%'})",
                },
                title=f"District mean: {INDICATOR_LABELS[indicator]}",
            )
            figure.update_layout(showlegend=selected_district != DISTRICT_ALL)
            st.plotly_chart(figure, width="stretch")
        st.caption(
            "District comparison values are means over selected months. "
            "The district control highlights the selected district; comparisons retain all districts."
        )

    st.markdown("**Detected month-over-month relative changes**")
    if filtered_trends.empty:
        st.info("No detected trends match the current filters.")
    else:
        trend_chart_data = filtered_trends.copy()
        trend_chart_data["district_month"] = (
            trend_chart_data["district"].astype(str)
            + " - "
            + _month_keys(trend_chart_data["month"])
        )
        trend_chart_data["indicator_label"] = trend_chart_data["indicator"].map(INDICATOR_LABELS)
        figure = px.bar(
            trend_chart_data,
            x="district_month",
            y="percentage_change",
            color="direction",
            facet_row="indicator_label",
            hover_data=["district", "month", "current_value", "absolute_change"],
            color_discrete_map={"increase": "#268b78", "decrease": "#cf6658"},
            labels={
                "district_month": "District and month",
                "percentage_change": "Relative change (%)",
                "indicator_label": "Indicator",
                "direction": "Direction",
            },
            title="Detected month-over-month changes",
        )
        figure.update_layout(height=max(420, 280 * trend_chart_data["indicator"].nunique()))
        figure.update_xaxes(tickangle=-45)
        st.plotly_chart(figure, width="stretch")

    st.markdown("**Full-population IQR box plots**")
    st.caption(
        "Outlier status and box-plot distributions use the full validated dataset, "
        "so district/month display filters do not alter the IQR population."
    )
    if selected_indicators:
        try:
            boxplot_images = _outlier_boxplot_images(data, outlier_results)
            for indicator in selected_indicators:
                image_name = f"outliers_box_{indicator}.png"
                if image_name in boxplot_images:
                    st.image(
                        boxplot_images[image_name],
                        caption=f"{INDICATOR_LABELS[indicator]} distribution, all districts and months",
                    )
        except Exception as error:
            st.error(f"Could not generate outlier box plots: {error}")
    else:
        st.info("Select one or more indicators to display box plots.")

with downloads_tab:
    st.subheader("Download reports")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Filtered insights CSV",
        data=_csv_bytes(filtered_insights),
        file_name="filtered_insights.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[1].download_button(
        "All insights CSV",
        data=_csv_bytes(all_insights),
        file_name="insights.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[0].download_button(
        "Correlation matrix CSV",
        data=correlation_matrix.to_csv(index_label="indicator").encode("utf-8"),
        file_name="correlation_matrix.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[1].download_button(
        "Structured insights JSON",
        data=all_insights.to_json(orient="records", indent=2),
        file_name="sample_insights.json",
        mime="application/json",
        width="stretch",
    )
    st.caption("Correlation matrix and all-insights downloads use the full validated dataset and current correlation threshold.")
