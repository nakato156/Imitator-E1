# v126 / Imitator Token-ID Prototype

Los experimentos v126/v126b se configuran por argumentos CLI en vez de TOML.
Esta familia es el primer Imitator real: video/keypoints de dataset1 ->
**token IDs de Gemma** -> texto decodificado/prompt de correccion v125.

La clasificacion aislada de 64 glosas se usa solo para inicializar el encoder
visual ST-GCN desde v121; no es la salida oficial de Imitator.

Level A, clip aislado -> token IDs de la glosa:

```bash
PYTHONPATH=. python scripts/train/train_imitator_dataset1_tokens.py
```

Level B, secuencias sinteticas dataset1 -> secuencia de token IDs:

```bash
PYTHONPATH=. python scripts/train/train_temporal_v126.py \
  --phase teacher_forced \
  --min-clips 2 \
  --max-clips 8 \
  --prediction-alpha-mode teacher_alpha
```

Continuation learned-CIF controlado desde un checkpoint teacher-forced:

```bash
PYTHONPATH=. python scripts/train/train_temporal_v126.py \
  --phase learned_cif \
  --resume ../outputs/v126_temporal/<teacher_run>/checkpoint_best.pt \
  --resume-weights-only \
  --diag-freeze target_only_stage1 \
  --alpha-schedule linear_pred_mix \
  --mix-w-pred-start 0.02 \
  --mix-w-pred-end 0.10 \
  --mix-ramp-epochs 20 \
  --prediction-alpha-mode pred_rescaled_to_target_len
```

Cada corrida escribe `predictions.jsonl` con `target_token_ids`,
`predicted_token_ids`, `target_text`, `predicted_text` y `gemma_prompt`.
