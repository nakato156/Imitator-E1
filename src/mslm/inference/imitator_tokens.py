"""Utilities for the true Imitator prototype: video -> Gemma token IDs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


GEMMA_V125_FEWSHOT_PROMPT = """Responde sólo con la salida, sin explicación.

Ejemplo 1
Entrada: se intentará que que la distancia mínima entre sombrillas y reposeras cumpla con la separación establecida
Salida: se intentará que que la distancia mínima entre sombrillas y reposeras cumpla con la separación establecida.

Ejemplo 2
Entrada: en mi caso trabajo en estados unidos y con comunidades sordas de otros países doy charlas dicto clases y hago otras actividades
Salida: en mi caso trabajo en estados unidos y con comunidades sordas de otros países, doy charlas, dicto clases y hago otras actividades.

Ejemplo 3
Entrada: claro porque si tiene acceso a esto luego pueden decidir y tener la autonomía de votar con información
Salida: claro porque si tiene acceso a esto luego pueden decidir y tener la autonomía de votar con información.

Entrada: {sequence}
Salida:"""


@dataclass(frozen=True)
class TokenPrediction:
    """Serializable prediction row for paper-prototype reports."""

    clip_ids: tuple[str, ...]
    glosses: tuple[str, ...]
    target_token_ids: tuple[int, ...]
    predicted_token_ids: tuple[int, ...]
    target_text: str
    predicted_text: str
    gemma_prompt: str

    def as_dict(self) -> dict:
        return {
            "clip_ids": list(self.clip_ids),
            "glosses": list(self.glosses),
            "target_token_ids": list(self.target_token_ids),
            "predicted_token_ids": list(self.predicted_token_ids),
            "target_text": self.target_text,
            "predicted_text": self.predicted_text,
            "gemma_prompt": self.gemma_prompt,
        }


def build_gemma_correction_prompt(sequence: str) -> str:
    """Return the v125 few-shot correction prompt for a decoded token sequence."""
    return GEMMA_V125_FEWSHOT_PROMPT.format(sequence=sequence.strip())


def strip_padding_token_ids(ids: Sequence[int], pad_id: int = -100) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in ids if int(token_id) != pad_id)


def align_logits_to_targets(logits: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or crop sequence logits to target length for deterministic decoding."""
    if logits.size(0) < target_steps:
        pad = logits.new_zeros((target_steps - logits.size(0), logits.size(-1)))
        logits = torch.cat([logits, pad], dim=0)
    elif logits.size(0) > target_steps:
        logits = logits[:target_steps]
    return logits


def make_token_predictions(
    *,
    token_logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_lengths: torch.Tensor,
    clip_ids: Iterable[Sequence[str]],
    glosses: Iterable[Sequence[str]],
    tokenizer,
) -> list[TokenPrediction]:
    """Convert model logits into token-id/text rows.

    The prototype uses the known target length for deterministic reporting. CIF
    learned-count quality is still measured separately by the trainer.
    """
    rows: list[TokenPrediction] = []
    clip_id_rows = [tuple(str(x) for x in row) for row in clip_ids]
    gloss_rows = [tuple(str(x) for x in row) for row in glosses]
    pred_ids = token_logits.argmax(dim=-1).detach().cpu()
    target_ids = token_ids.detach().cpu()
    lengths = token_lengths.detach().cpu().tolist()
    for i, length in enumerate(lengths):
        target = strip_padding_token_ids(target_ids[i, :length].tolist())
        pred = tuple(int(x) for x in pred_ids[i, :length].tolist())
        target_text = tokenizer.decode(target, skip_special_tokens=True)
        predicted_text = tokenizer.decode(pred, skip_special_tokens=True)
        rows.append(
            TokenPrediction(
                clip_ids=clip_id_rows[i],
                glosses=gloss_rows[i],
                target_token_ids=target,
                predicted_token_ids=pred,
                target_text=target_text,
                predicted_text=predicted_text,
                gemma_prompt=build_gemma_correction_prompt(predicted_text),
            )
        )
    return rows
