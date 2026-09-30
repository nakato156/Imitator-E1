import torch

from src.mslm.inference.imitator_tokens import (
    build_gemma_correction_prompt,
    make_token_predictions,
    strip_padding_token_ids,
)


class TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(token_id)}" for token_id in ids)


def test_strip_padding_token_ids_removes_training_pad():
    assert strip_padding_token_ids([7, -100, 8, -100]) == (7, 8)


def test_build_gemma_correction_prompt_uses_v125_fewshot_shape():
    prompt = build_gemma_correction_prompt("hola mundo")

    assert "Responde sólo con la salida" in prompt
    assert "Entrada: hola mundo" in prompt
    assert prompt.endswith("Salida:")


def test_make_token_predictions_exports_ids_text_and_prompt():
    logits = torch.zeros(1, 3, 12)
    logits[0, 0, 4] = 3.0
    logits[0, 1, 5] = 3.0
    logits[0, 2, 6] = 3.0
    rows = make_token_predictions(
        token_logits=logits,
        token_ids=torch.tensor([[1, 2, -100]]),
        token_lengths=torch.tensor([2]),
        clip_ids=[("9",)],
        glosses=[("hola",)],
        tokenizer=TinyTokenizer(),
    )

    assert len(rows) == 1
    row = rows[0].as_dict()
    assert row["clip_ids"] == ["9"]
    assert row["glosses"] == ["hola"]
    assert row["target_token_ids"] == [1, 2]
    assert row["predicted_token_ids"] == [4, 5]
    assert row["target_text"] == "tok1 tok2"
    assert row["predicted_text"] == "tok4 tok5"
    assert "Entrada: tok4 tok5" in row["gemma_prompt"]
