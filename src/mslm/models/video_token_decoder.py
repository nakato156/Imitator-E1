"""Autoregressive video-to-Gemma-token decoder, independent of CIF."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


BOS_ID = 2
EOS_ID = 106
PAD_ID = 0
GEMMA_VOCAB_SIZE = 262_400
MAX_CONTENT_TOKENS = 32
MAX_OUTPUT_TOKENS = MAX_CONTENT_TOKENS + 1  # content plus EOS


@dataclass(frozen=True)
class GreedyDecodeOutput:
    """Decoded content tokens (EOS excluded) and termination information."""

    token_ids: list[list[int]]
    emitted_eos: torch.Tensor
    lengths: torch.Tensor


def causal_mask(length: int, *, device=None) -> torch.Tensor:
    """Boolean Transformer mask: True positions are not visible."""
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


def build_teacher_forcing(
    token_ids: torch.Tensor,
    *,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
    source_padding_id: int = -100,
    max_content_tokens: int = MAX_CONTENT_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``[BOS, tokens] -> [tokens, EOS]`` without target lengths."""
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [B, L]")
    rows_in: list[torch.Tensor] = []
    rows_out: list[torch.Tensor] = []
    for row in token_ids:
        valid = row[row.ne(source_padding_id)]
        if valid.numel() > max_content_tokens:
            raise ValueError(
                f"target has {valid.numel()} tokens; maximum is {max_content_tokens}"
            )
        rows_in.append(torch.cat([row.new_tensor([bos_id]), valid]))
        rows_out.append(torch.cat([valid, row.new_tensor([eos_id])]))
    max_len = max(x.numel() for x in rows_in)
    decoder_input = token_ids.new_full((len(rows_in), max_len), pad_id)
    labels = token_ids.new_full((len(rows_out), max_len), pad_id)
    for i, (inputs, targets) in enumerate(zip(rows_in, rows_out)):
        decoder_input[i, : inputs.numel()] = inputs
        labels[i, : targets.numel()] = targets
    return decoder_input, labels


class VideoTokenDecoder(nn.Module):
    """ST-GCN frame encoder followed by a causal Transformer decoder.

    The token embedding and output projection are exactly the same parameter.
    No CIF quantities, boundaries, or target lengths are accepted by this API.

    ``vocab_map`` optionally restricts the output layer to the given Gemma token
    ids (a dense reparameterisation of the softmax: row *i* IS Gemma id
    ``vocab_map[i]``, so the model still consumes and emits real Gemma ids —
    ``greedy_decode`` returns them already remapped). ``bos_id``/``eos_id``/
    ``pad_id`` are always given as Gemma ids; internally the model works in
    dense ids when a map is present.

    ``with_ctc`` attaches an auxiliary frame-level CTC head over the same
    (restricted) vocabulary plus a final blank class at index ``vocab_size``.
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        *,
        hidden_size: int = 128,
        vocab_size: int = GEMMA_VOCAB_SIZE,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_size: int = 512,
        dropout: float = 0.1,
        max_content_tokens: int = MAX_CONTENT_TOKENS,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        pad_id: int = PAD_ID,
        vocab_map: Sequence[int] | None = None,
        with_ctc: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.frame_encoder = frame_encoder
        self.hidden_size = int(hidden_size)
        self.vocab_map = None
        if vocab_map is not None:
            self.vocab_map = [int(t) for t in vocab_map]
            if len(set(self.vocab_map)) != len(self.vocab_map):
                raise ValueError("vocab_map has duplicate Gemma ids")
            for special in (bos_id, eos_id, pad_id):
                if int(special) not in self.vocab_map:
                    raise ValueError(f"special Gemma id {special} missing from vocab_map")
            vocab_size = len(self.vocab_map)
            table = torch.full((GEMMA_VOCAB_SIZE,), -1, dtype=torch.long)
            table[torch.tensor(self.vocab_map)] = torch.arange(vocab_size)
            self.register_buffer("gemma_to_dense", table, persistent=False)
            self.register_buffer(
                "dense_to_gemma", torch.tensor(self.vocab_map), persistent=False
            )
            bos_id = self.vocab_map.index(int(bos_id))
            eos_id = self.vocab_map.index(int(eos_id))
            pad_id = self.vocab_map.index(int(pad_id))
        self.vocab_size = int(vocab_size)
        self.max_content_tokens = int(max_content_tokens)
        self.max_output_tokens = self.max_content_tokens + 1
        self.bos_id, self.eos_id, self.pad_id = int(bos_id), int(eos_id), int(pad_id)

        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(self.max_output_tokens, hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.ctc_head = nn.Linear(hidden_size, self.vocab_size + 1) if with_ctc else None
        self.ctc_blank_id = self.vocab_size if with_ctc else None

    def to_dense(self, gemma_ids: torch.Tensor) -> torch.Tensor:
        """Map Gemma ids to dense ids; negatives (e.g. -100) pass through."""
        if self.vocab_map is None:
            return gemma_ids
        negative = gemma_ids < 0
        dense = self.gemma_to_dense[gemma_ids.clamp(min=0)]
        if bool((dense[~negative] < 0).any()):
            bad = gemma_ids[~negative][dense[~negative] < 0]
            raise ValueError(f"Gemma ids not in vocab_map: {bad.unique().tolist()[:5]}")
        return dense.where(~negative, gemma_ids)

    def to_gemma(self, dense_ids: list[int]) -> list[int]:
        if self.vocab_map is None:
            return list(dense_ids)
        return [self.vocab_map[i] for i in dense_ids]

    @staticmethod
    def frame_padding_mask(frame_lengths: torch.Tensor, frame_count: int) -> torch.Tensor:
        steps = torch.arange(frame_count, device=frame_lengths.device)
        return steps.unsqueeze(0) >= frame_lengths.unsqueeze(1)

    def decode_features(
        self,
        frame_features: torch.Tensor,
        frame_lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        length = decoder_input_ids.size(1)
        if length > self.max_output_tokens:
            raise ValueError(
                f"decoder length {length} exceeds maximum {self.max_output_tokens}"
            )
        positions = torch.arange(length, device=decoder_input_ids.device)
        target = self.token_embedding(decoder_input_ids) + self.position_embedding(positions)
        decoded = self.decoder(
            target,
            frame_features,
            tgt_mask=causal_mask(length, device=target.device),
            tgt_key_padding_mask=decoder_input_ids.eq(self.pad_id),
            memory_key_padding_mask=self.frame_padding_mask(
                frame_lengths, frame_features.size(1)
            ),
        )
        return F.linear(self.final_norm(decoded), self.token_embedding.weight, self.output_bias)

    def forward(
        self,
        keypoints: torch.Tensor,
        frame_lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        features = self.frame_encoder(keypoints, frame_lengths)
        return self.decode_features(features, frame_lengths, decoder_input_ids)

    @torch.no_grad()
    def greedy_decode(
        self,
        keypoints: torch.Tensor,
        frame_lengths: torch.Tensor,
    ) -> GreedyDecodeOutput:
        """Greedy decode through first EOS; EOS is forbidden at step zero."""
        features = self.frame_encoder(keypoints, frame_lengths)
        batch_size = keypoints.size(0)
        inputs = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.long, device=keypoints.device
        )
        sequences: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=keypoints.device)
        lengths = torch.full(
            (batch_size,), self.max_output_tokens, dtype=torch.long, device=keypoints.device
        )

        for step in range(self.max_output_tokens):
            next_logits = self.decode_features(features, frame_lengths, inputs)[:, -1].clone()
            if step == 0:
                next_logits[:, self.eos_id] = -torch.inf
            next_ids = next_logits.argmax(dim=-1)
            for i, token in enumerate(next_ids.tolist()):
                if finished[i]:
                    continue
                if token == self.eos_id:
                    finished[i] = True
                    lengths[i] = len(sequences[i])
                else:
                    sequences[i].append(token)
            if bool(finished.all()):
                break
            # Finished rows receive PAD and are masked; unfinished rows extend normally.
            appended = next_ids.masked_fill(finished, self.pad_id)
            inputs = torch.cat([inputs, appended.unsqueeze(1)], dim=1)

        return GreedyDecodeOutput(
            [self.to_gemma(seq) for seq in sequences], finished, lengths
        )


def set_decoder_training_stage(
    model: VideoTokenDecoder,
    epoch: int,
    *,
    unfreeze_stgcn_epoch: int | None = None,
) -> dict[str, bool]:
    """Epochs 0-4 decoder only; epochs 5+ also visual TCN/Transformer.

    ``unfreeze_stgcn_epoch`` additionally unfreezes ST-GCN + linear_hidden from
    that epoch on (E3 variant); ``None`` keeps them frozen forever (default).
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    decoder_modules = [
        model.token_embedding,
        model.position_embedding,
        model.decoder,
        model.final_norm,
    ]
    if model.ctc_head is not None:
        decoder_modules.append(model.ctc_head)
    for module in decoder_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.output_bias.requires_grad = True
    visual_enabled = epoch >= 5
    for name in ("tcn", "transformer"):
        module = getattr(model.frame_encoder, name)
        for parameter in module.parameters():
            parameter.requires_grad = visual_enabled
    stgcn_enabled = unfreeze_stgcn_epoch is not None and epoch >= unfreeze_stgcn_epoch
    for name in ("stgcn_layers", "linear_hidden"):
        module = getattr(model.frame_encoder, name)
        for parameter in module.parameters():
            parameter.requires_grad = stgcn_enabled
    return {
        "decoder": True,
        "frame_encoder.tcn": visual_enabled,
        "frame_encoder.transformer": visual_enabled,
        "frame_encoder.stgcn_layers": stgcn_enabled,
        "frame_encoder.linear_hidden": stgcn_enabled,
    }


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_lineage(state: dict, outer_signer: int) -> dict:
    lineage = state.get("lineage")
    if not lineage:
        raise RuntimeError("source checkpoint has no lineage")
    contaminated = set(lineage.get("train_signers", ())) | set(
        lineage.get("val_signers", ())
    )
    if outer_signer in contaminated:
        raise RuntimeError(
            f"outer signer {outer_signer} contaminates source checkpoint lineage"
        )
    recorded_outer = lineage.get("fold_outer_test_signer")
    if recorded_outer is not None and int(recorded_outer) != int(outer_signer):
        raise RuntimeError(
            f"checkpoint outer signer is {recorded_outer}, expected {outer_signer}"
        )
    return lineage


def initialize_from_cif_checkpoint(
    model: VideoTokenDecoder,
    checkpoint_path: str | Path,
    *,
    outer_signer: int,
    expected_sha256: str | None = None,
) -> dict:
    """Load only frame encoder and token classifier weight/bias from CIF."""
    actual_hash = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    state = torch.load(checkpoint_path, map_location="cpu")
    lineage = validate_checkpoint_lineage(state, outer_signer)
    source = state.get("model", state)

    encoder_state = {
        name.removeprefix("frame_encoder."): value
        for name, value in source.items()
        if name.startswith("frame_encoder.")
    }
    missing, unexpected = model.frame_encoder.load_state_dict(encoder_state, strict=True)
    if missing or unexpected:  # strict=True normally raises; kept explicit for test doubles.
        raise RuntimeError(f"incomplete frame encoder load: {missing=}, {unexpected=}")

    classifier_weight = source.get("token_head.classifier.weight")
    classifier_bias = source.get("token_head.classifier.bias")
    if classifier_weight is None or classifier_bias is None:
        raise RuntimeError("source checkpoint lacks token_head classifier parameters")
    if model.vocab_map is not None:
        # Restricted vocabulary: copy only the rows of the mapped Gemma ids.
        rows = model.dense_to_gemma.to(classifier_weight.device)
        classifier_weight = classifier_weight[rows]
        classifier_bias = classifier_bias[rows]
    if classifier_weight.shape != model.token_embedding.weight.shape:
        raise RuntimeError(
            f"classifier shape {tuple(classifier_weight.shape)} != "
            f"embedding shape {tuple(model.token_embedding.weight.shape)}"
        )
    if classifier_bias.shape != model.output_bias.shape:
        raise RuntimeError("classifier bias shape does not match output bias")
    with torch.no_grad():
        model.token_embedding.weight.copy_(classifier_weight)
        model.output_bias.copy_(classifier_bias)
    return {"checkpoint_sha256": actual_hash, "lineage": lineage}
