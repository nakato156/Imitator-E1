"""Recomputa val_ce / accuracy reales del checkpoint v115.1 para localizar el bug
de la métrica Loss/val_ce (reportada 0.568, imposible: el piso con label_smoothing=0.1
sobre V=262400 es 1.248 y una predicción perfecta da 2.68).

Carga el checkpoint best, reproduce el split de validación (seed 42) y computa la CE de
varias formas, imprimiendo el vocabulario efectivo y el término de smoothing.
"""
from settings import initialize
initialize()

import torch
import torch.nn.functional as F

import importlib.util
from torch.utils.data import DataLoader
from src.mslm.utils.setup_train import prepare_datasets, build_model
from src.mslm.dataloader import collate_fn, BatchSampler
from src.mslm.utils.paths import path_vars

_spec = importlib.util.spec_from_file_location("lce", "src/mslm/training/loss_ce_vocab.py")
_lce = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_lce)
imitator_ce_loss = _lce.imitator_ce_loss   # la MISMA función que usa el Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "outputs/checkpoints/1151/1/best/checkpoint.pth"
EMBED_TABLE = "../data/processed/gemma3n_embed_table.pt"
LOG_TEMP = 2.74          # ~ep18-23 (TB); temp = exp(2.74) ~ 15.5
LABEL_SMOOTHING = 0.1
IGNORE_INDEX = -100

model_cfg = dict(input_size=111, output_size=2048, hidden_size=1024, nhead=16,
                 ff_dim=2816, n_layers=6, encoder_dropout=0.45,
                 multihead_dropout=0.4, pool_dim=256)

def main():
    # --- modelo ---
    model = build_model(**model_cfg).to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    print(f"checkpoint epoch={ck['epoch']} | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # --- tabla de embeddings (igual que el Trainer: float + normalizada) ---
    E = torch.load(EMBED_TABLE, map_location="cpu").float()
    En = F.normalize(E, dim=-1)
    V = En.shape[0]
    temp = torch.tensor(LOG_TEMP).exp().item()
    print(f"V={V} | temp=exp({LOG_TEMP})={temp:.3f} | piso_ls={LABEL_SMOOTHING*torch.log(torch.tensor(float(V))):.4f}")

    # --- val loader (mismo split seed 42) ---
    _, val_ds, _, _ = prepare_datasets(path_vars.h5_file, 0.8, 111,
                                       include_datasets=["dataset2"], return_token_ids=True)
    # batch=64 como el run real (BatchSampler agrupa por longitud); sub_batch=4 interno.
    val_dl = DataLoader(val_ds, num_workers=2, collate_fn=collate_fn,
                        batch_sampler=BatchSampler(val_ds, 64))
    log_temp_t = torch.tensor(LOG_TEMP)
    SUB = 4

    En_gpu = En.to(DEVICE)                 # tabla en GPU para reproducir el path con autocast
    log_temp_g = log_temp_t.to(DEVICE)
    ce_fp32, ce_bf16 = [], []              # CE por sub-batch en fp32 vs autocast-bf16

    with torch.no_grad():
        for batch in val_dl:
            keypoint, fpm, emb, mask_emb = batch[:4]
            ids = batch[4].to(DEVICE)
            B = keypoint.size(0)
            for i in range(0, B, SUB):
                s, e = i, min(i + SUB, B)
                kp = keypoint[s:e].to(DEVICE); fp = fpm[s:e].to(DEVICE)
                tid = ids[s:e]

                # fp32 (referencia)
                out, _ = model(kp, fp)
                L = min(out.size(1), tid.size(1))
                ce32, _, _, _ = imitator_ce_loss(out[:, :L].float(), tid[:, :L],
                                                 En_gpu, log_temp_g, LABEL_SMOOTHING)
                ce_fp32.append(float(ce32))

                # autocast bf16 — EXACTAMENTE como el Trainer (with accelerator.autocast())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out2, _ = model(kp, fp)
                    ceb, _, _, _ = imitator_ce_loss(out2[:, :L], tid[:, :L],
                                                    En_gpu, log_temp_g, LABEL_SMOOTHING)
                ce_bf16.append(float(ceb))
                del out, out2; torch.cuda.empty_cache()

    import statistics as st
    print(f"\nsub-batches evaluados: {len(ce_fp32)}\n")
    print("=== CE media por sub-batch (label_smoothing=0.1) ===")
    print(f"  fp32          : {st.mean(ce_fp32):.4f}")
    print(f"  autocast bf16 : {st.mean(ce_bf16):.4f}   <-- modo del Trainer")
    print(f"\n  >>> TensorBoard reportó Loss/val_ce ~= 0.568 (ep18-23)")
    print(f"  piso teórico ls=0.1 sobre V={V}: {LABEL_SMOOTHING*torch.log(torch.tensor(float(V))):.4f}")


if __name__ == "__main__":
    main()
