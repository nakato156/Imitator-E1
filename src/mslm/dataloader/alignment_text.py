"""Normalización de texto usada para unidades visuales de v125."""

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")


def normalize_alignment_text(text: str) -> str:
    """NFC, minúsculas, sin puntuación y con espacios normalizados."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return _SPACE_RE.sub(" ", text).strip()
