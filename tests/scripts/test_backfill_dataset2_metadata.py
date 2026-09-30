import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))
from backfill_dataset2_metadata import compute_metadata_fields  # noqa: E402


def test_computes_expected_fields():
    fields = compute_metadata_fields(
        fname="lsa-noticias-en-lengua-de-senas-argentina-resumen-semanal-13062021_227.mp4",
        video_column="lsa-noticias-en-lengua-de-senas-argentina-resumen-semanal-13062021",
        label="Sí, de a poco lo vamos consiguiendo.",
        frame_count=74,
        token_count=11,
    )
    assert fields["video_id"] == "lsa-noticias-en-lengua-de-senas-argentina-resumen-semanal-13062021_227"
    assert fields["source_group"] == "lsa-noticias-en-lengua-de-senas-argentina-resumen-semanal-13062021"
    assert fields["frame_count"] == 74
    assert fields["token_count"] == 11
    assert fields["word_count"] == 7  # "sí de a poco lo vamos consiguiendo" tokenizado
