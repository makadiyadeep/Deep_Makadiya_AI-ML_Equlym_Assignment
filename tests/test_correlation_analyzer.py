import warnings

import numpy as np
import pandas as pd
import pytest

from src.correlation_analyzer import INDICATORS, analyze_correlations


def make_frame(
    anc=None,
    institutional=None,
    immunization=None,
    high_risk=None,
):
    columns = [anc, institutional, immunization, high_risk]
    length = next((len(column) for column in columns if column is not None), 6)
    baseline = list(range(20, 20 + length))
    values = [column if column is not None else baseline for column in columns]
    return pd.DataFrame(dict(zip(INDICATORS, values)))


def analyze_quietly(data, threshold=0.70):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return analyze_correlations(data, threshold=threshold)


def test_detects_strong_positive_correlation():
    values = [1, 2, 3, 4, 5, 6]
    data = make_frame(anc=values, institutional=[value * 2 for value in values])

    matrix, relationships = analyze_quietly(data)

    assert matrix.loc["anc_coverage", "institutional_delivery"] == pytest.approx(1.0)
    pair = relationships.loc[
        (relationships.indicator_1 == "anc_coverage")
        & (relationships.indicator_2 == "institutional_delivery")
    ].iloc[0]
    assert pair.relationship == "positive"
    assert pair.strength == "strong"


def test_detects_strong_negative_correlation():
    values = [1, 2, 3, 4, 5, 6]
    data = make_frame(anc=values, institutional=list(reversed(values)))

    matrix, relationships = analyze_quietly(data)

    assert matrix.loc["anc_coverage", "institutional_delivery"] == pytest.approx(-1.0)
    pair = relationships.loc[
        (relationships.indicator_1 == "anc_coverage")
        & (relationships.indicator_2 == "institutional_delivery")
    ].iloc[0]
    assert pair.relationship == "negative"
    assert pair.strength == "strong"


def test_weak_correlation_is_not_reported_at_default_threshold():
    data = make_frame(anc=[1, 2, 3, 4, 5, 6], institutional=[2, 6, 3, 5, 1, 4])

    matrix, relationships = analyze_quietly(data)

    coefficient = matrix.loc["anc_coverage", "institutional_delivery"]
    assert abs(coefficient) < 0.30
    pair_found = (
        (relationships.indicator_1 == "anc_coverage")
        & (relationships.indicator_2 == "institutional_delivery")
    ).any()
    assert not pair_found


def test_threshold_filters_relationships_configurably():
    data = make_frame(anc=[1, 2, 3, 4, 5, 6], institutional=[2, 6, 3, 5, 1, 4])

    high_threshold_matrix, high_threshold_pairs = analyze_quietly(data, threshold=0.70)
    low_threshold_matrix, low_threshold_pairs = analyze_quietly(data, threshold=0.05)
    pair_mask = lambda pairs: (
        (pairs.indicator_1 == "anc_coverage")
        & (pairs.indicator_2 == "institutional_delivery")
    )

    assert not pair_mask(high_threshold_pairs).any()
    assert pair_mask(low_threshold_pairs).any()
    assert high_threshold_matrix.equals(low_threshold_matrix)


def test_pairs_are_unique_and_never_self_correlations():
    values = [1, 2, 3, 4, 5, 6]
    data = make_frame(anc=values, institutional=values, immunization=values, high_risk=values)

    _, relationships = analyze_quietly(data)

    assert all(relationships.indicator_1 != relationships.indicator_2)
    pairs = list(zip(relationships.indicator_1, relationships.indicator_2))
    assert len(pairs) == len(set(pairs))
    assert len(pairs) == 6


def test_constant_columns_are_undefined_and_reported():
    data = make_frame(anc=[7, 7, 7, 7, 7, 7])

    with pytest.warns(UserWarning, match="Constant-valued indicators"):
        matrix, relationships = analyze_correlations(data)

    assert matrix["anc_coverage"].isna().all()
    assert "anc_coverage" not in relationships.indicator_1.tolist()
    assert "anc_coverage" not in relationships.indicator_2.tolist()


def test_missing_invalid_and_nonfinite_values_are_excluded_pairwise():
    data = make_frame(
        anc=[1, 2, None, 4, 5, "invalid", np.inf],
        institutional=[2, 4, 6, 8, 10, 12, 14],
    )

    with pytest.warns(UserWarning, match="excluded pairwise"):
        matrix, _ = analyze_correlations(data)

    assert matrix.loc["anc_coverage", "institutional_delivery"] == pytest.approx(1.0)


def test_empty_and_insufficient_inputs_return_undefined_correlations():
    empty = pd.DataFrame(columns=INDICATORS)
    with pytest.warns(UserWarning, match="fewer than"):
        empty_matrix, empty_pairs = analyze_correlations(empty)
    assert empty_matrix.isna().all().all()
    assert empty_pairs.empty

    insufficient = make_frame(anc=[1, 2], institutional=[2, 4])
    with pytest.warns(UserWarning, match="at least 3 pairwise observations"):
        small_matrix, small_pairs = analyze_correlations(insufficient)
    assert pd.isna(small_matrix.loc["anc_coverage", "institutional_delivery"])
    assert small_pairs.empty


def test_warns_when_pairwise_sample_is_small_in_a_large_dataset():
    anc = list(range(1, 11)) + [None] * 30
    institutional = [value * 2 for value in range(1, 11)] + [None] * 30
    data = make_frame(anc=anc, institutional=institutional)

    with pytest.warns(UserWarning, match="Pairwise sample sizes below 30"):
        matrix, _ = analyze_correlations(data)

    assert matrix.loc["anc_coverage", "institutional_delivery"] == pytest.approx(1.0)
