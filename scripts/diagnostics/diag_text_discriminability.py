"""Opción 1 del plan post-diagnóstico v118: ¿es el LADO TEXTO discriminable?

Sin GPU, sin entrenar nada: si los 600 embeddings de frase (mean-pool de los
embeddings de entrada de Gemma, igual que `ContrastiveAligner.encode_text`) ya
son casi indistinguibles entre sí -- o hay oraciones duplicadas -- entonces
ningún encoder de vídeo, sin importar su calidad, puede alcanzar R@1 alto: el
techo de retrieval ya está limitado por el propio lado texto, no por el
encoder. Esto decide si vale la pena seguir iterando en la rama contrastiva
(v118) antes de invertir en cambios de pooling o en v119.

Uso:
    PYTHONPATH=. python scripts/diagnostics/diag_text_discriminability.py
"""
import json
import h5py
import numpy as np
import torch

from src.mslm.utils.paths import path_vars

INCLUDE_DATASETS = ["dataset2"]


def load_texts_and_labels(h5_path):
    pooled, labels, clip_ids = [], [], []
    with h5py.File(h5_path, "r") as f:
        for ds in INCLUDE_DATASETS:
            clips = sorted(f[ds]["embeddings"].keys())
            for clip in clips:
                emb = f[ds]["embeddings"][clip][:]            # [T_text, 2048]
                lab = f[ds]["labels"][clip][:][0].decode()
                pooled.append(emb.mean(axis=0))
                labels.append(lab)
                clip_ids.append(f"{ds}/{clip}")
    return torch.tensor(np.stack(pooled), dtype=torch.float32), labels, clip_ids


def main():
    h5_path = str(path_vars.h5_file)
    T, labels, clip_ids = load_texts_and_labels(h5_path)
    N = T.size(0)
    print(f"[diag-text] N={N} frases cargadas de {INCLUDE_DATASETS}")

    # --- 1. Duplicados exactos de texto ---
    from collections import Counter
    counts = Counter(labels)
    dups = {k: v for k, v in counts.items() if v > 1}
    n_dup_clips = sum(dups.values())
    print(f"\n=== Duplicados de etiqueta ===")
    print(f"  Frases únicas: {len(counts)} / {N} clips")
    print(f"  Frases repetidas: {len(dups)} grupos, {n_dup_clips} clips afectados "
          f"({100*n_dup_clips/N:.1f}%)")
    if dups:
        sample = list(dups.items())[:5]
        for text, cnt in sample:
            print(f"    x{cnt}: {text[:80]!r}")

    # --- 2. Similitud coseno entre embeddings de texto (mean-pooled, sin normalizar) ---
    Tn = torch.nn.functional.normalize(T, dim=-1)
    sim = Tn @ Tn.T                                            # [N, N]
    off_diag_mask = ~torch.eye(N, dtype=torch.bool)
    off_vals = sim[off_diag_mask]

    print(f"\n=== Similitud coseno entre TODAS las parejas texto-texto (fuera de diagonal) ===")
    print(f"  media={off_vals.mean():.4f}  std={off_vals.std():.4f}  "
          f"p50={off_vals.median():.4f}  p99={off_vals.quantile(0.99):.4f}  max={off_vals.max():.4f}")

    # --- 3. Vecino más cercano (excluyendo self) por frase ---
    sim_masked = sim.clone()
    sim_masked.fill_diagonal_(-1.0)
    nn_sim, nn_idx = sim_masked.max(dim=1)
    print(f"\n=== Vecino más cercano (texto-a-texto, excluyendo self) ===")
    print(f"  cos medio={nn_sim.mean():.4f}  mediana={nn_sim.median():.4f}  "
          f"max={nn_sim.max():.4f}  min={nn_sim.min():.4f}")
    for thr in (0.90, 0.95, 0.99, 0.999):
        frac = (nn_sim >= thr).float().mean().item()
        print(f"  frac con vecino_más_cercano >= {thr}: {frac:.1%}")

    # casos más extremos (excluyendo duplicados exactos ya contados arriba)
    order = torch.argsort(nn_sim, descending=True)
    print(f"\n  Top 5 pares más cercanos (no necesariamente idénticos):")
    seen = set()
    shown = 0
    for i in order.tolist():
        j = nn_idx[i].item()
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        print(f"    cos={nn_sim[i]:.4f} | {clip_ids[i]} {labels[i][:50]!r}  <->  "
              f"{clip_ids[j]} {labels[j][:50]!r}")
        shown += 1
        if shown >= 5:
            break

    # --- 4. Rank efectivo de los embeddings de texto (anisotropía) ---
    Tc = T - T.mean(dim=0, keepdim=True)
    cov = (Tc.T @ Tc) / (N - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = eigvals / eigvals.sum()
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum()
    eff_rank = entropy.exp().item()
    print(f"\n=== Rank efectivo de los embeddings de texto (de {T.size(1)} dims) ===")
    print(f"  effective_rank={eff_rank:.2f}  (cuanto más bajo, más anisotrópico/concentrado el espacio)")

    # --- 5. Veredicto ---
    print(f"\n=== VEREDICTO ===")
    frac_095 = (nn_sim >= 0.95).float().mean().item()
    if n_dup_clips / N > 0.05 or frac_095 > 0.10:
        print("-> El LADO TEXTO tiene un techo de retrieval intrínsecamente bajo:")
        print(f"   {100*n_dup_clips/N:.1f}% de clips comparten frase exacta y/o "
              f"{frac_095:.1%} tienen un vecino con cos>=0.95.")
        print("   Ningún encoder de vídeo puede superar este techo sin resolver la ambigüedad")
        print("   textual (deduplicar, usar sentence-embeddings reales en vez de mean-pool de")
        print("   input-embeddings, o aceptar que R@1 exacto no es la métrica adecuada).")
    else:
        print("-> El lado texto es razonablemente discriminable (pocos duplicados/vecinos")
        print("   cercanos). El techo de R@1~3% observado en v118 NO se explica por")
        print("   ambigüedad textual -> el cuello de botella está más probablemente en el")
        print("   pooling/arquitectura del encoder de vídeo o en la dificultad genuina de la")
        print("   tarea con 481 pares de entrenamiento.")

    out = {
        "N": N,
        "n_unique_labels": len(counts),
        "n_dup_clips": n_dup_clips,
        "pairwise_cos_mean": off_vals.mean().item(),
        "pairwise_cos_p99": off_vals.quantile(0.99).item(),
        "nn_cos_mean": nn_sim.mean().item(),
        "nn_cos_median": nn_sim.median().item(),
        "frac_nn_cos_ge_095": frac_095,
        "frac_nn_cos_ge_099": (nn_sim >= 0.99).float().mean().item(),
        "text_effective_rank": eff_rank,
    }
    with open("outputs/diag_text_discriminability.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[diag-text] Resultados guardados en outputs/diag_text_discriminability.json")


if __name__ == "__main__":
    main()
