"""Tests para greedy_ctc_decode y word_error_rate (v119 — CTC sobre secuencia).

WER = (S+D+I)/N (ecuación 9 de LiftSign, CVPRW 2026), a nivel de palabra.
"""
import importlib.util
from pathlib import Path

import torch

# Carga directa del archivo (ver test_vocab.py): evita depender del orden de
# ejecución de la suite frente a la polución de sys.modules que otros tests
# (test_contrastive_aligner.py) hacen sobre "src"/"src.mslm".
_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("wer", _ROOT / "src/mslm/utils/wer.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
greedy_ctc_decode = _mod.greedy_ctc_decode
word_error_rate = _mod.word_error_rate


def _log_probs_from_ids(id_seq: list[int], vocab_size: int) -> torch.Tensor:
    """Construye un tensor [T, V] donde argmax en cada paso t es id_seq[t]."""
    T = len(id_seq)
    logits = torch.full((T, vocab_size), -10.0)
    for t, idx in enumerate(id_seq):
        logits[t, idx] = 10.0
    return torch.log_softmax(logits, dim=-1)


def test_greedy_decode_collapses_consecutive_repeats():
    log_probs = _log_probs_from_ids([1, 1, 2, 2, 3], vocab_size=4).unsqueeze(0)
    lengths = torch.tensor([5])
    decoded = greedy_ctc_decode(log_probs, lengths, blank=0)
    assert decoded == [[1, 2, 3]]


def test_greedy_decode_removes_blanks():
    log_probs = _log_probs_from_ids([0, 1, 0, 2, 0], vocab_size=4).unsqueeze(0)
    lengths = torch.tensor([5])
    decoded = greedy_ctc_decode(log_probs, lengths, blank=0)
    assert decoded == [[1, 2]]


def test_greedy_decode_mixed_repeats_and_blanks():
    log_probs = _log_probs_from_ids([1, 1, 0, 2, 2, 0, 3], vocab_size=4).unsqueeze(0)
    lengths = torch.tensor([7])
    decoded = greedy_ctc_decode(log_probs, lengths, blank=0)
    assert decoded == [[1, 2, 3]]


def test_greedy_decode_respects_lengths_padding():
    # Después de longitud=3, hay basura de padding que debe ignorarse.
    log_probs = _log_probs_from_ids([1, 2, 3, 1, 1, 1], vocab_size=4).unsqueeze(0)
    lengths = torch.tensor([3])
    decoded = greedy_ctc_decode(log_probs, lengths, blank=0)
    assert decoded == [[1, 2, 3]]


def test_greedy_decode_batch_with_different_lengths():
    a = _log_probs_from_ids([1, 1, 0, 2], vocab_size=4)
    b = _log_probs_from_ids([3, 0, 0, 0], vocab_size=4)
    log_probs = torch.stack([a, b])
    lengths = torch.tensor([4, 1])
    decoded = greedy_ctc_decode(log_probs, lengths, blank=0)
    assert decoded == [[1, 2], [3]]


def test_word_error_rate_exact_match():
    assert word_error_rate(["hola", "mundo"], ["hola", "mundo"]) == 0.0


def test_word_error_rate_one_substitution():
    assert word_error_rate(["hola", "tierra"], ["hola", "mundo"]) == 0.5


def test_word_error_rate_one_deletion():
    # ref tiene 3 palabras, hyp le falta una -> 1 deletion / 3
    assert word_error_rate(["a", "c"], ["a", "b", "c"]) == 1 / 3


def test_word_error_rate_one_insertion():
    # ref tiene 2 palabras, hyp tiene una de más -> 1 insertion / 2
    assert word_error_rate(["a", "b", "c"], ["a", "b"]) == 0.5
