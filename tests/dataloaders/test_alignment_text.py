import unicodedata

from src.mslm.dataloader.alignment_text import normalize_alignment_text


def test_normalize_alignment_text_applies_closed_v125_rules():
    decomposed = unicodedata.normalize("NFD", "DÍAS")
    text = f"  ¡Buenos,\t{decomposed}!  "
    assert normalize_alignment_text(text) == "buenos días"


def test_normalize_alignment_text_replaces_punctuation_with_space():
    assert normalize_alignment_text("señas/lenguaje—visual") == "señas lenguaje visual"
