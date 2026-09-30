"""Métricas de texto para el oracle de Gemma y el grounded_score de v128
(v125). ROUGE-L se implementa a mano (LCS) para no añadir `rouge_score`+
`nltk` solo por una métrica; chrF/BLEU usan `sacrebleu` (sin deps pesadas)."""
import sacrebleu

from src.mslm.dataloader.vocab import tokenize

_SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "a", "al",
    "en", "y", "o", "que", "se", "su", "sus", "lo", "le", "les", "es", "son",
    "por", "para", "con", "no", "sí", "como", "más", "pero", "ya", "muy",
}


def rouge_l_f1(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            if p[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def content_word_f1(pred: str, ref: str) -> float:
    p = {w for w in tokenize(pred) if w not in _SPANISH_STOPWORDS}
    r = {w for w in tokenize(ref) if w not in _SPANISH_STOPWORDS}
    if not p or not r:
        return 0.0
    overlap = len(p & r)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def chrf_score(pred: str, ref: str) -> float:
    return sacrebleu.sentence_chrf(pred, [ref]).score


def bleu_score(pred: str, ref: str) -> float:
    return sacrebleu.sentence_bleu(pred, [ref]).score
