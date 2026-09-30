"""Probing de activaciones por capa para CTCEncoder (v119) -- diferencia entre
dos hipótesis sobre la causa raíz del colapso a blank visto en
`diagnose_ctc_posteriors.py` (P(blank) domina todo el interior de la
secuencia, la única señal no-blank aparece en los bordes):

  (a) el BiLSTM no logra propagar información discriminativa de frames
      intermedios (la señal existe a su entrada pero se pierde dentro de él).
  (b) la señal que llega desde GCN+TCN/TLP a la entrada del BiLSTM ya viene
      plana por frame (el BiLSTM solo refleja sus propios estados de borde).

Técnica: en 4 puntos del forward de `CTCEncoder` (gcn, tcn1, tcn2_short,
bilstm_long), mide cuánta variación temporal frame-a-frame sobrevive, vía
`activity_c = var_t(x_c) / (mean_t(x_c**2) + eps)` por canal, promediada sobre
canales. Score cercano a 0 = representación casi constante en el tiempo
(plana); más alto = más variación frame a frame. Ver
the experiment configuration and command-line options.

Reconstrucción manual del forward (sin forward hooks, los submódulos de
CTCEncoder son atributos públicos) -- no modifica el modelo productivo.

Uso (mismo patrón que diagnose_ctc_posteriors.py, debe correr desde la raíz
del repo):
    MSLM_EXPERIMENT_CONFIG=experiments/v119_ctc/ctc_v119.toml \
        PYTHONPATH=. python scripts/diagnostics/diagnose_ctc_activations.py --n-clips 20
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_ctc_posteriors import build_val

STAGES = ["gcn", "tcn1", "tcn2_short", "bilstm_long"]

# (label, tag, run_dir) -- run_dir=None usa str(run_id) del config, como siempre.
CHECKPOINTS = [
    ("v119/1 (data-oriented levers)", "best_wer", None),
    ("v119/1_baseline_no_data_levers", "best_wer", "1_baseline_no_data_levers"),
]


def probe_forward(model, x: torch.Tensor, frames_padding_mask: torch.Tensor) -> dict:
    """Replica CTCEncoder.forward() capturando el tensor [B, hidden, T_etapa]
    en cada uno de los 4 puntos de sondeo. x: [B, T, N, 2]."""
    lengths = (~frames_padding_mask).sum(dim=1)

    static_in = x.permute(0, 3, 1, 2)  # [B, 2, T, N]
    feats = model._run_stack(model.stgcn_layers, static_in)
    if model.use_motion_stream:
        motion_in = model._motion_stream(x).permute(0, 3, 1, 2)
        feats_m = model._run_stack(model.stgcn_motion_layers, motion_in)
        feats = torch.cat([feats, feats_m], dim=1)

    feats = model.linear_hidden(feats)  # [B, hidden, T, N]
    gcn = feats.mean(dim=-1)            # [B, hidden, T]

    feats = F.relu(model.tcn_conv1(gcn))
    feats_bt = feats.permute(0, 2, 1).contiguous()  # [B, T, hidden]
    feats_bt, lengths, _ = model.tlp1(feats_bt, lengths)
    tcn1 = feats_bt.permute(0, 2, 1).contiguous()  # [B, hidden, T1]

    feats = F.relu(model.tcn_conv2(tcn1))
    feats_bt = feats.permute(0, 2, 1).contiguous()
    feats_short, lengths, _ = model.tlp2(feats_bt, lengths)  # [B, T2, hidden]
    tcn2_short = feats_short.permute(0, 2, 1).contiguous()   # [B, hidden, T2]

    packed = pack_padded_sequence(
        feats_short, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
    )
    packed_out, _ = model.bilstm(packed)
    feats_long, _ = pad_packed_sequence(
        packed_out, batch_first=True, total_length=feats_short.size(1)
    )
    bilstm_long = feats_long.permute(0, 2, 1).contiguous()  # [B, hidden, T2]

    return {"gcn": gcn, "tcn1": tcn1, "tcn2_short": tcn2_short, "bilstm_long": bilstm_long}


def temporal_activity(x: torch.Tensor) -> float:
    """x: [B, hidden, T]. var_t/power_t por canal, promediada sobre canales."""
    var = x.var(dim=2, unbiased=False)
    power = (x**2).mean(dim=2)
    return float((var / (power + 1e-8)).mean())


def run_clip(model, keypoint, frames_mask, device):
    keypoint = keypoint.to(device)
    frames_mask = frames_mask.to(device)
    with torch.no_grad():
        probes = probe_forward(model, keypoint, frames_mask)
    return [temporal_activity(probes[stage]) for stage in STAGES]


def plot_scores(scores: np.ndarray, out_dir: str):
    x = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in scores:
        ax.plot(x, row, color="tab:blue", alpha=0.2, linewidth=1)
    ax.plot(x, scores.mean(axis=0), color="tab:red", linewidth=2.5, label="media sobre clips")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel("actividad temporal (var/power)")
    ax.set_title("Actividad temporal por etapa del forward")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/activity_by_stage.png", dpi=120)
    plt.close(fig)


def write_summary(scores: np.ndarray, out_dir: str) -> str:
    lines = [f"Clips evaluados: {len(scores)}"]
    for i, stage in enumerate(STAGES):
        lines.append(f"{stage}: {scores[:, i].mean():.4f} ± {scores[:, i].std():.4f}")
    text = "\n".join(lines)
    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write(text + "\n")
    return text


def diagnose_checkpoint(label, tag, run_dir, n_clips) -> np.ndarray:
    # strict=False: el probing no llega a `classifier`, así que un vocab_size
    # desalineado entre checkpoint y vocab.json (ver docstring de
    # _load_state_dict_skip_mismatched) no afecta las activaciones medidas.
    model, val_dl, _vocab, device, diag_out_dir = build_val(tag, run_dir=run_dir, strict=False)
    out_dir = f"{diag_out_dir}/activations"
    os.makedirs(out_dir, exist_ok=True)

    scores = []
    for i, (keypoint, frames_mask, *_rest) in enumerate(val_dl):
        if i >= n_clips:
            break
        scores.append(run_clip(model, keypoint, frames_mask, device))
    scores = np.array(scores)

    plot_scores(scores, out_dir)
    text = write_summary(scores, out_dir)
    print(f"\n[diag-act] {label} ({out_dir}):\n{text}")
    return scores


def main(args):
    results = {}
    for label, tag, run_dir in CHECKPOINTS:
        results[label] = diagnose_checkpoint(label, tag, run_dir, args.n_clips)

    print("\n[diag-act] Comparación entre checkpoints (media ± std por etapa):")
    print("etapa".ljust(14) + "".join(label.ljust(38) for label in results))
    for i, stage in enumerate(STAGES):
        row = stage.ljust(14)
        for scores in results.values():
            row += f"{scores[:, i].mean():.4f} ± {scores[:, i].std():.4f}".ljust(38)
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=20, help="Clips de val a sondear por checkpoint.")
    main(parser.parse_args())
