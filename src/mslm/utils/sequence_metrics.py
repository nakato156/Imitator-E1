"""Strict token-sequence metrics for the A3/CIF Etapa 4 correction.

``legacy_prefix_exact`` is the historical metric: it only compares the first
``len(target)`` predicted tokens, so an over/under-predicted sequence whose
prefix happens to match still counts as exact. ``strict_exact`` additionally
requires the predicted token count to equal the target length, per the
Etapa 4 correction plan.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Mapping, Sequence


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def token_error_rate(pred: Sequence[int], target: Sequence[int]) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, target) / len(target)


def token_edit_similarity(ter: float) -> float:
    return max(0.0, 1.0 - ter)


def legacy_prefix_exact(pred: Sequence[int], target: Sequence[int]) -> bool:
    """Historical metric: prefix of length len(target) must match. Ignores count."""
    return list(pred[: len(target)]) == list(target)


def strict_exact(pred: Sequence[int], target: Sequence[int]) -> bool:
    """pred_count == target_len AND every token matches."""
    return len(pred) == len(target) and list(pred) == list(target)


def parse_gloss_tokens(
    token_ids: Sequence[int], token_ids_by_gloss: Mapping[str, Sequence[int]]
) -> tuple[list[str], list[int]]:
    """Greedily parse concatenated Gemma ids into glosses, longest code first.

    This is the versioned equivalent of the audit's Fase2b parser.  Returning
    the unparsed suffix makes parse failures explicit instead of silently
    dropping tokens.
    """
    code_to_gloss: dict[tuple[int, ...], str] = {}
    for gloss, code in token_ids_by_gloss.items():
        normalized = tuple(map(int, code))
        if not normalized:
            raise ValueError(f"gloss {gloss!r} has an empty token code")
        previous = code_to_gloss.setdefault(normalized, str(gloss))
        if previous != str(gloss):
            raise ValueError(
                f"glosses {previous!r} and {gloss!r} share token code {normalized}"
            )

    ordered_codes = sorted(code_to_gloss, key=lambda code: (-len(code), code))
    ids = tuple(map(int, token_ids))
    parsed: list[str] = []
    cursor = 0
    while cursor < len(ids):
        for code in ordered_codes:
            if ids[cursor : cursor + len(code)] == code:
                parsed.append(code_to_gloss[code])
                cursor += len(code)
                break
        else:
            return parsed, list(ids[cursor:])
    return parsed, []


def pairwise_gloss_order_counts(
    predicted_glosses: Sequence[str], target_glosses: Sequence[str]
) -> tuple[int, int]:
    """Count concordant relative-order pairs among unambiguous common glosses.

    A gloss is eligible only when it occurs exactly once in both sequences,
    matching the original Fase2b audit definition.  Samples with fewer than
    two eligible glosses contribute zero pairs and are therefore excluded from
    the denominator.
    """
    predicted = list(predicted_glosses)
    target = list(target_glosses)
    predicted_counts = Counter(predicted)
    target_counts = Counter(target)
    common = sorted(
        gloss
        for gloss in set(predicted) & set(target)
        if predicted_counts[gloss] == 1 and target_counts[gloss] == 1
    )
    if len(common) < 2:
        return 0, 0

    predicted_position = {gloss: predicted.index(gloss) for gloss in common}
    target_position = {gloss: target.index(gloss) for gloss in common}
    correct = sum(
        (predicted_position[a] < predicted_position[b])
        == (target_position[a] < target_position[b])
        for a, b in combinations(common, 2)
    )
    total = len(common) * (len(common) - 1) // 2
    return int(correct), total


def gloss_sequence_diagnostics(
    predictions: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    token_ids_by_gloss: Mapping[str, Sequence[int]],
) -> dict[str, float | int | None]:
    """Aggregate the reproducible Fase2b gloss diagnostics over paired rows."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")

    prediction_parseable = target_parseable = multiset_exact = 0
    precision_numerator = precision_denominator = 0
    recall_numerator = recall_denominator = 0
    order_correct = order_total = order_eligible_samples = 0

    for prediction, target in zip(predictions, targets):
        predicted_glosses, predicted_leftover = parse_gloss_tokens(
            prediction, token_ids_by_gloss
        )
        target_glosses, target_leftover = parse_gloss_tokens(target, token_ids_by_gloss)
        prediction_parseable += not predicted_leftover
        target_parseable += not target_leftover

        predicted_counts = Counter(predicted_glosses)
        target_counts = Counter(target_glosses)
        intersection = sum((predicted_counts & target_counts).values())
        precision_numerator += intersection
        precision_denominator += max(1, len(predicted_glosses))
        recall_numerator += intersection
        recall_denominator += len(target_glosses)
        multiset_exact += predicted_counts == target_counts and not (
            predicted_leftover or target_leftover
        )

        correct, total = pairwise_gloss_order_counts(predicted_glosses, target_glosses)
        order_correct += correct
        order_total += total
        order_eligible_samples += total > 0

    sample_count = len(predictions)
    return {
        "sample_count": sample_count,
        "prediction_parse_rate": prediction_parseable / sample_count if sample_count else None,
        "target_parse_rate": target_parseable / sample_count if sample_count else None,
        "gloss_precision": (
            precision_numerator / precision_denominator if precision_denominator else None
        ),
        "gloss_recall": recall_numerator / recall_denominator if recall_denominator else None,
        "multiset_exact": multiset_exact / sample_count if sample_count else None,
        "pairwise_order_accuracy": order_correct / order_total if order_total else None,
        "pairwise_order_correct": order_correct,
        "pairwise_order_total": order_total,
        "pairwise_order_eligible_samples": order_eligible_samples,
    }
