from src.mslm.utils.sequence_metrics import (
    gloss_sequence_diagnostics,
    legacy_prefix_exact,
    levenshtein,
    pairwise_gloss_order_counts,
    parse_gloss_tokens,
    strict_exact,
    token_edit_similarity,
    token_error_rate,
)


def test_strict_exact_requires_count_match():
    assert strict_exact([1, 2], [1, 2]) is True
    assert strict_exact([1, 2, 3], [1, 2]) is False  # overprediction
    assert strict_exact([1], [1, 2]) is False  # underprediction
    assert strict_exact([1, 3], [1, 2]) is False  # wrong token


def test_legacy_prefix_exact_ignores_extra_predicted_tokens():
    # Historical bug this plan fixes: a 3-token overprediction whose prefix
    # matches the 2-token target still counted as "exact".
    assert legacy_prefix_exact([1, 2, 9], [1, 2]) is True
    assert strict_exact([1, 2, 9], [1, 2]) is False


def test_token_error_rate_and_similarity():
    assert token_error_rate([1, 2], [1, 2]) == 0.0
    assert token_edit_similarity(0.0) == 1.0
    ter = token_error_rate([1, 3], [1, 2])
    assert ter == 0.5
    assert token_edit_similarity(ter) == 0.5


def test_token_error_rate_empty_target():
    assert token_error_rate([], []) == 0.0
    assert token_error_rate([1], []) == 1.0


def test_levenshtein_basic():
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [1, 2]) == 1
    assert levenshtein([], [1, 2]) == 2


def test_parse_gloss_tokens_uses_longest_match_and_returns_leftover():
    codes = {"Leche": [7], "Leche dulce": [7, 8], "Agua": [9]}
    assert parse_gloss_tokens([7, 8, 9], codes) == (["Leche dulce", "Agua"], [])
    assert parse_gloss_tokens([9, 99], codes) == (["Agua"], [99])


def test_pairwise_gloss_order_reproduces_fase2b_unique_occurrence_rule():
    assert pairwise_gloss_order_counts(["B", "A", "C"], ["A", "B", "C"]) == (2, 3)
    # Repeated A is ambiguous and excluded; B/C is the only eligible pair.
    assert pairwise_gloss_order_counts(
        ["A", "B", "A", "C"], ["A", "C", "A", "B"]
    ) == (0, 1)
    assert pairwise_gloss_order_counts(["A"], ["A", "B"]) == (0, 0)


def test_gloss_diagnostics_excludes_rows_with_fewer_than_two_common_glosses():
    diagnostics = gloss_sequence_diagnostics(
        predictions=[[2, 1], [1]],
        targets=[[1, 2], [1, 3]],
        token_ids_by_gloss={"A": [1], "B": [2], "C": [3]},
    )
    assert diagnostics["pairwise_order_accuracy"] == 0.0
    assert diagnostics["pairwise_order_total"] == 1
    assert diagnostics["pairwise_order_eligible_samples"] == 1
    assert diagnostics["gloss_precision"] == 1.0
    assert diagnostics["gloss_recall"] == 3 / 4


if __name__ == "__main__":
    test_strict_exact_requires_count_match()
    test_legacy_prefix_exact_ignores_extra_predicted_tokens()
    test_token_error_rate_and_similarity()
    test_token_error_rate_empty_target()
    test_levenshtein_basic()
    print("ok")
