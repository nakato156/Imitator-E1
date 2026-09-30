#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
echo "[$(date -Is)] starting autoregressive paper run: folds 4-6"
for fold in 4 5 6; do
  echo "[$(date -Is)] fold=${fold} CIF comparator"
  python scripts/train/train_video_token_decoder.py cif-comparator --fold "$fold"
  echo "[$(date -Is)] fold=${fold} autoregressive training"
  python scripts/train/train_video_token_decoder.py train --fold "$fold"
  echo "[$(date -Is)] fold=${fold} closed-checkpoint outer evaluation"
  python scripts/train/train_video_token_decoder.py evaluate \
    --fold "$fold" \
    --checkpoint "../outputs/video_token_decoder/fold${fold}_seed23/base/checkpoint_closed.pt" \
    --cif-comparator "../outputs/video_token_decoder/fold${fold}_cif_comparator.json"
done
echo "[$(date -Is)] autoregressive paper run completed"
