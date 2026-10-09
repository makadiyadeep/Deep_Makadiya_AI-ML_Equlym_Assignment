"""Calculate pairwise Pearson correlations and threshold-qualified pairs."""

import warnings

import numpy as np
import pandas as pd

INDICATORS = (
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
)
INDICATOR_LABELS = {
    "anc_coverage": "ANC coverage",
    "institutional_delivery": "Institutional delivery",
    "immunization": "Immunization",
    "high_risk_cases": "High-risk cases",
}
RELATIONSHIP_COLUMNS = (
    "indicator_1",
    "indicator_2",
    "correlation",
    "relationship",
    "strength",
)
MIN_PAIR_OBSERVATIONS = 3
SMALL_SAMPLE_THRESHOLD = 30


def correlation_strength(coefficient: float) -> str:
    """Classify magnitude only; these labels do not indicate significance."""
    magnitude = abs(float(coefficient))
    if magnitude >= 0.70:
        return "strong"
    if magnitude >= 0.30:
        return "moderate"
    return "weak"


def analyze_correlations(
    df: pd.DataFrame,
    threshold: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a Pearson matrix and unique pairs meeting ``threshold``.

    Correlations use pairwise-complete finite numeric observations. Pairs with
    fewer than three observations remain undefined; datasets with fewer than
    30 rows are flagged as potentially unstable. Bounds, p-values, and causal
    interpretations are outside this analyzer's scope.
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be a number in the interval (0, 1]") from error
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be a number in the interval (0, 1]")

    missing_columns = [column for column in INDICATORS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required indicators: {missing_columns}")

    numeric = pd.DataFrame(index=df.index)
    excluded_values: list[str] = []
    for indicator in INDICATORS:
        original = df[indicator]
        converted = pd.to_numeric(original, errors="coerce").astype(float)
        invalid_conversion = original.notna() & converted.isna()
        non_finite = converted.notna() & ~np.isfinite(converted.to_numpy())
        missing_count = int(original.isna().sum())
        invalid_count = int(invalid_conversion.sum() + non_finite.sum())
        if missing_count:
            excluded_values.append(f"{indicator}: {missing_count} missing")
        if invalid_count:
            excluded_values.append(f"{indicator}: {invalid_count} invalid/non-finite")
        numeric[indicator] = converted.mask(non_finite)

    if excluded_values:
        warnings.warn(
            "Missing, invalid, or non-finite numerical values were excluded "
            "pairwise: " + "; ".join(excluded_values),
            UserWarning,
            stacklevel=2,
        )
    if len(df) < SMALL_SAMPLE_THRESHOLD:
        warnings.warn(
            f"Only {len(df)} rows are available; correlations may be unstable "
            f"with fewer than {SMALL_SAMPLE_THRESHOLD} observations.",
            UserWarning,
            stacklevel=2,
        )

    constant_indicators = [
        indicator
        for indicator in INDICATORS
        if numeric[indicator].nunique(dropna=True) == 1
    ]
    if constant_indicators:
        warnings.warn(
            "Constant-valued indicators have undefined Pearson correlations: "
            + ", ".join(constant_indicators),
            UserWarning,
            stacklevel=2,
        )

    insufficient_pairs = []
    unstable_pairs = []
    for first_index, indicator_1 in enumerate(INDICATORS):
        for indicator_2 in INDICATORS[first_index + 1 :]:
            pair_count = int(numeric[[indicator_1, indicator_2]].dropna().shape[0])
            if pair_count < MIN_PAIR_OBSERVATIONS:
                insufficient_pairs.append(
                    f"{indicator_1}/{indicator_2} (n={pair_count})"
                )
            elif pair_count < SMALL_SAMPLE_THRESHOLD:
                unstable_pairs.append(
                    f"{indicator_1}/{indicator_2} (n={pair_count})"
                )
    if insufficient_pairs:
        warnings.warn(
            f"Pearson correlations require at least {MIN_PAIR_OBSERVATIONS} "
            "pairwise observations; these pairs are undefined: "
            + ", ".join(insufficient_pairs),
            UserWarning,
            stacklevel=2,
        )
    if unstable_pairs and len(df) >= SMALL_SAMPLE_THRESHOLD:
        warnings.warn(
            f"Pairwise sample sizes below {SMALL_SAMPLE_THRESHOLD} may make "
            "these correlations unstable: " + ", ".join(unstable_pairs),
            UserWarning,
            stacklevel=2,
        )

    correlation_matrix = numeric.corr(
        method="pearson",
        min_periods=MIN_PAIR_OBSERVATIONS,
    ).reindex(index=INDICATORS, columns=INDICATORS)

    relationships: list[dict[str, object]] = []
    for first_index, indicator_1 in enumerate(INDICATORS):
        for indicator_2 in INDICATORS[first_index + 1 :]:
            coefficient = correlation_matrix.at[indicator_1, indicator_2]
            if pd.isna(coefficient) or abs(float(coefficient)) < threshold:
                continue
            coefficient = float(coefficient)
            relationships.append(
                {
                    "indicator_1": indicator_1,
                    "indicator_2": indicator_2,
                    "correlation": coefficient,
                    "relationship": "positive" if coefficient > 0 else "negative",
                    "strength": correlation_strength(coefficient),
                }
            )

    return correlation_matrix, pd.DataFrame(
        relationships,
        columns=RELATIONSHIP_COLUMNS,
    )
