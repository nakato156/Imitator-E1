"""Run Gemma v125 correction over Imitator prediction rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.mslm.models.gemma_bridge import GemmaBridge


DEFAULT_MODEL = (
    ".hf_cache/hub/"
    "models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/"
    "snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or args.predictions.with_suffix(".gemma_corrected.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bridge = GemmaBridge(args.model_id, device=device)
    rows = []
    with args.predictions.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if args.limit is not None and len(rows) >= args.limit:
                break

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = row["gemma_prompt"]
            encoded = bridge.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            generated = bridge.generate_from_ids(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                attention_mask=attention_mask,
            )
            corrected = bridge.decode(generated)[0].strip()
            row["gemma_corrected_text"] = corrected
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[gemma-correct] wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
