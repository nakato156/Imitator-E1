# Video → tokens Gemma (decoder autoregresivo)

Esta ruta es independiente de CIF. Usa el vocabulario Gemma completo, carga
solamente `frame_encoder` y el clasificador token-level de cada checkpoint
`foldN_promotion`, y nunca entrega boundaries, alphas ni target lengths al
modelo.

Comandos por fold:

```bash
# Comparador CIF; ajusta a,b en inner-val y los congela para outer-test.
python scripts/train/train_video_token_decoder.py cif-comparator --fold 1

# Entrena 15 epochs y cierra el mejor checkpoint por
# (strict_exact, token_edit_similarity) de inner-val.
python scripts/train/train_video_token_decoder.py train --fold 1

# Outer-test sólo acepta checkpoint_closed.pt.
python scripts/train/train_video_token_decoder.py evaluate --fold 1 \
  --checkpoint ../outputs/video_token_decoder/fold1_seed23/base/checkpoint_closed.pt \
  --cif-comparator ../outputs/video_token_decoder/fold1_cif_comparator.json
```

El protocolo completo y reanudable (CIF 1–6, desarrollo 1–3, única variante
de rescate, freeze y confirmación 4–6) se ejecuta con:

```bash
python scripts/train/run_video_token_decoder_protocol.py
```

Sólo si quedan al menos cinco horas y no se adoptó rescate:

```bash
python scripts/train/run_video_token_decoder_protocol.py --remaining-hours 5
```

Los scripts rechazan folds 7–10, hashes de manifiesto/checkpoint inválidos,
lineage contaminado y evaluación outer de checkpoints sin cerrar.
