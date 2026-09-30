"""Filtro de integridad para clips de dataset2 (v125).

Aplica las reglas de exclusión del plan v125-v129 en un orden fijo y
determinista: cada clip recibe exactamente una causa (la primera regla que
viola), nunca una lista de causas. El manifiesto cubre TODOS los clips de
entrada (kept=True incluido) para que sea auditable sin cruzar con otra
fuente.
"""
from dataclasses import dataclass

MIN_FRAMES = 8
MIN_VISUAL_TOKENS = 2
MIN_FRAMES_PER_TOKEN = 2.0


@dataclass
class ClipRecord:
    clip_id: str
    has_keypoints: bool
    has_label: bool
    has_embeddings: bool
    has_token_ids: bool
    frame_count: int
    token_count: int
    embedding_rows: int
    truncated: bool
    keypoints_have_nan_inf: bool
    embeddings_have_nan_inf: bool


def _cause_for(rec: ClipRecord) -> str | None:
    if not rec.has_keypoints:
        return "missing_keypoints"
    if not rec.has_label:
        return "missing_label"
    if not rec.has_embeddings:
        return "missing_embeddings"
    if not rec.has_token_ids:
        return "missing_token_ids"
    if rec.frame_count < MIN_FRAMES:
        return "min_frames"
    if rec.token_count < MIN_VISUAL_TOKENS:
        return "min_tokens"
    if rec.token_count > 0 and rec.frame_count / rec.token_count < MIN_FRAMES_PER_TOKEN:
        return "frames_per_token"
    if rec.truncated:
        return "truncated"
    if rec.token_count != rec.embedding_rows:
        return "token_embedding_mismatch"
    if rec.keypoints_have_nan_inf:
        return "nan_inf_keypoints"
    if rec.embeddings_have_nan_inf:
        return "nan_inf_embeddings"
    return None


def filter_clip_records(records: list[ClipRecord]) -> tuple[list[str], list[dict]]:
    kept: list[str] = []
    manifest: list[dict] = []
    for rec in records:
        cause = _cause_for(rec)
        manifest.append({"clip_id": rec.clip_id, "kept": cause is None, "cause": cause})
        if cause is None:
            kept.append(rec.clip_id)
    return kept, manifest
