"""Decodificación greedy de CTC + Word Error Rate (v119).

WER = (S+D+I)/N (ecuación 9, LiftSign CVPRW 2026) — la métrica de evaluación
estándar de la literatura cargada para CSLR, a nivel de palabra.
"""
import torch


def greedy_ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """Argmax + colapso de repetidos consecutivos + remoción de blanks.

    log_probs: [B, T, V] (log-probabilidades, solo se usa el argmax).
    lengths:   [B] longitud válida (no-padding) de cada secuencia.
    """
    B = log_probs.size(0)
    ids = log_probs.argmax(dim=-1)  # [B, T]

    decoded = []
    for b in range(B):
        seq = ids[b, : lengths[b].item()].tolist()
        out = []
        prev = None
        for tok in seq:
            if tok != blank and tok != prev:
                out.append(tok)
            prev = tok
        decoded.append(out)
    return decoded


def word_error_rate(hyp_words: list[str], ref_words: list[str]) -> float:
    """Distancia de Levenshtein a nivel palabra entre hyp y ref, normalizada por len(ref)."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n
