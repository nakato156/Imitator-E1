import pytest

from src.mslm.utils.text_metrics import bleu_score, chrf_score, content_word_f1, rouge_l_f1


def test_rouge_l_identical_strings_is_one():
    assert rouge_l_f1("el gato come pescado", "el gato come pescado") == pytest.approx(1.0)


def test_rouge_l_disjoint_strings_is_zero():
    assert rouge_l_f1("el gato come pescado", "xyz abc def ghi") == pytest.approx(0.0)


def test_rouge_l_partial_overlap_in_between():
    score = rouge_l_f1("el gato come pescado", "el perro come carne")
    assert 0.0 < score < 1.0


def test_content_word_f1_ignores_stopwords():
    # difieren solo en stopwords -> F1 de palabras de contenido debe ser 1.0
    assert content_word_f1("el gato come el pescado", "un gato come pescado") == pytest.approx(1.0)


def test_content_word_f1_empty_prediction_is_zero():
    assert content_word_f1("", "el gato come pescado") == 0.0


def test_chrf_identical_strings_is_near_100():
    assert chrf_score("buenos días a todos", "buenos días a todos") > 99.0


def test_chrf_unrelated_strings_is_low():
    assert chrf_score("buenos días a todos", "x y z") < 20.0


def test_bleu_identical_strings_is_near_100():
    assert bleu_score("buenos días a todos los presentes", "buenos días a todos los presentes") > 99.0
