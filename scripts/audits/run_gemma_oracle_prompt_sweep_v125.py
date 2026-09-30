"""Barrido de prompts para el oracle Gemma v125.

El objetivo no es crear un artefacto final, sino diagnosticar si el gate de
chrF>=70 falla por prompt/formato de instrucción o por el modelo/target.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.mslm.utils  # noqa: E402,F401
from src.mslm.models.gemma_bridge import GemmaBridge  # noqa: E402
from src.mslm.utils.text_metrics import (  # noqa: E402
    bleu_score,
    chrf_score,
    content_word_f1,
    rouge_l_f1,
)

MODEL_ID = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"


@dataclass(frozen=True)
class PromptVariant:
    name: str
    pre: str
    post: str
    add_bos: bool = True


@dataclass
class OracleSample:
    clip_id: str
    label: str
    token_ids: np.ndarray
    embeddings: np.ndarray


BASE_PROMPTS = [
    PromptVariant(
        "plan_original",
        "Traduce la siguiente representación de lengua de señas a español natural.\n",
        "\nRespuesta:",
    ),
    PromptVariant(
        "directo_tokens",
        "Traduce a español natural la secuencia de señas:\n",
        "\nTraducción:",
    ),
    PromptVariant(
        "lsa_directo",
        "La siguiente secuencia corresponde a lengua de señas argentina. Escríbela en español natural:\n",
        "\nEspañol:",
    ),
    PromptVariant(
        "imperativo_corto",
        "Convierte esta secuencia de señas en una oración en español:\n",
        "\n",
    ),
    PromptVariant(
        "solo_respuesta",
        "Secuencia de señas:\n",
        "\nRespuesta en español:",
    ),
    PromptVariant(
        "sin_contexto_minimo",
        "",
        "\n",
    ),
    PromptVariant(
        "chat_user_assistant",
        "<start_of_turn>user\nTraduce esta secuencia de lengua de señas a español natural:\n",
        "\n<end_of_turn>\n<start_of_turn>model\n",
    ),
    PromptVariant(
        "chat_lsa",
        "<start_of_turn>user\nLa secuencia es de lengua de señas argentina. Da sólo la traducción en español natural:\n",
        "\n<end_of_turn>\n<start_of_turn>model\n",
    ),
]


STRICT_PROMPTS = [
    PromptVariant(
        "strict_lsa_directo",
        (
            "La siguiente secuencia corresponde a lengua de señas argentina.\n"
            "Escribe únicamente la traducción final en español natural.\n"
            "No expliques. No uses markdown. No agregues notas. No uses comillas.\n"
            "Secuencia:\n"
        ),
        "\nTraducción:",
    ),
    PromptVariant(
        "strict_solo_oracion",
        (
            "Tarea: traducir lengua de señas argentina a español natural.\n"
            "Respuesta obligatoria: una sola oración o frase final.\n"
            "Prohibido: explicaciones, listas, markdown, código, notas o comentarios.\n"
            "Entrada:\n"
        ),
        "\nSalida:",
    ),
    PromptVariant(
        "strict_no_extra",
        (
            "Convierte esta secuencia de señas en español natural.\n"
            "Devuelve sólo el texto traducido, nada antes ni después.\n"
        ),
        "\n",
    ),
    PromptVariant(
        "strict_chat",
        (
            "<start_of_turn>user\n"
            "Traduce la secuencia de lengua de señas argentina a español natural.\n"
            "Responde sólo con la traducción. No expliques, no uses markdown, "
            "no agregues notas.\n"
        ),
        "\n<end_of_turn>\n<start_of_turn>model\n",
    ),
]


def load_samples(
    h5_path: Path, split_path: Path, n_samples: int, seed: int
) -> list[OracleSample]:
    split = json.loads(split_path.read_text())
    rng = random.Random(seed)
    selected = rng.sample(split["val"], min(n_samples, len(split["val"])))
    samples = []
    with h5py.File(h5_path, "r") as file:
        dataset = file["dataset2"]
        for clip_id in selected:
            samples.append(
                OracleSample(
                    clip_id=clip_id,
                    label=dataset["labels"][clip_id][0].decode(),
                    token_ids=dataset["token_ids"][clip_id][:].astype(np.int64),
                    embeddings=dataset["embeddings"][clip_id][:].astype(np.float32),
                )
            )
    return samples


def load_train_examples(
    h5_path: Path, split_path: Path, n_examples: int, seed: int
) -> list[OracleSample]:
    if n_examples <= 0:
        return []
    split = json.loads(split_path.read_text())
    rng = random.Random(seed + 1009)
    selected = rng.sample(split["train"], min(n_examples, len(split["train"])))
    examples = []
    with h5py.File(h5_path, "r") as file:
        dataset = file["dataset2"]
        for clip_id in selected:
            examples.append(
                OracleSample(
                    clip_id=clip_id,
                    label=dataset["labels"][clip_id][0].decode(),
                    token_ids=dataset["token_ids"][clip_id][:].astype(np.int64),
                    embeddings=dataset["embeddings"][clip_id][:].astype(np.float32),
                )
            )
    return examples


def build_fewshot_prompts(
    bridge: GemmaBridge, examples: list[OracleSample]
) -> list[PromptVariant]:
    if not examples:
        return []

    blocks = []
    for idx, example in enumerate(examples, start=1):
        text = bridge.tokenizer.decode(example.token_ids, skip_special_tokens=True)
        blocks.append(
            f"Ejemplo {idx}\n"
            f"Entrada: {text}\n"
            f"Salida: {example.label}\n"
        )
    examples_text = "\n".join(blocks)
    return [
        PromptVariant(
            f"fewshot_{len(examples)}_strict_lsa",
            (
                "Aprende el formato de respuesta de los ejemplos.\n"
                "La tarea es traducir lengua de señas argentina a español natural.\n"
                "Responde únicamente con la traducción final.\n"
                "No expliques. No uses markdown. No agregues notas, código ni comentarios.\n\n"
                f"{examples_text}\n"
                "Ahora traduce esta entrada.\n"
                "Entrada: "
            ),
            "\nSalida:",
        ),
        PromptVariant(
            f"fewshot_{len(examples)}_minimal",
            (
                "Responde sólo con la salida, sin explicación.\n\n"
                f"{examples_text}\n"
                "Entrada: "
            ),
            "\nSalida:",
        ),
    ]


def select_prompts(
    prompt_set: str, bridge: GemmaBridge, examples: list[OracleSample]
) -> list[PromptVariant]:
    if prompt_set == "base":
        return BASE_PROMPTS
    if prompt_set == "strict":
        return STRICT_PROMPTS + build_fewshot_prompts(bridge, examples)
    if prompt_set == "all":
        return BASE_PROMPTS + STRICT_PROMPTS + build_fewshot_prompts(bridge, examples)
    raise ValueError(f"prompt_set desconocido: {prompt_set}")


def prompt_token_ids(
    bridge: GemmaBridge, variant: PromptVariant, sign_ids: torch.Tensor
) -> torch.Tensor:
    pre = bridge.tokenizer(
        variant.pre, return_tensors="pt", add_special_tokens=variant.add_bos
    ).input_ids.to(sign_ids.device)
    post = bridge.tokenizer(
        variant.post, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(sign_ids.device)
    return torch.cat([pre, sign_ids.unsqueeze(0), post], dim=1)


def prompt_embeddings(
    bridge: GemmaBridge, variant: PromptVariant, sign_embeddings: torch.Tensor
) -> torch.Tensor:
    device = sign_embeddings.device
    pre_ids = bridge.tokenizer(
        variant.pre, return_tensors="pt", add_special_tokens=variant.add_bos
    ).input_ids.to(device)
    post_ids = bridge.tokenizer(
        variant.post, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    return torch.cat(
        [
            bridge.embed_tokens(pre_ids),
            sign_embeddings.unsqueeze(0),
            bridge.embed_tokens(post_ids),
        ],
        dim=1,
    )


def summarize(rows: list[dict], key: str = "prediction") -> dict:
    def average(metric):
        return sum(metric(row[key], row["label"]) for row in rows) / max(len(rows), 1)

    return {
        "chrf": average(chrf_score),
        "bleu": average(bleu_score),
        "rouge_l": average(rouge_l_f1),
        "content_word_f1": average(content_word_f1),
    }


def run_variant(
    bridge: GemmaBridge,
    samples: list[OracleSample],
    variant: PromptVariant,
    max_new_tokens: int,
    check_exact_embeddings: bool,
) -> dict:
    device = next(bridge.lm.parameters()).device
    rows = []
    exact_rows = []
    for index, sample in enumerate(samples, start=1):
        sign_ids = torch.from_numpy(sample.token_ids).to(device=device, dtype=torch.long)
        prompt_ids = prompt_token_ids(bridge, variant, sign_ids)
        output_ids = bridge.generate_from_ids(prompt_ids, max_new_tokens=max_new_tokens)
        prediction = bridge.decode(output_ids)[0].strip()
        row = {
            "clip_id": sample.clip_id,
            "label": sample.label,
            "prediction": prediction,
            "token_count": int(len(sample.token_ids)),
        }
        rows.append(row)

        if check_exact_embeddings:
            live = bridge.embed_tokens(sign_ids.unsqueeze(0))[0]
            stored = torch.from_numpy(sample.embeddings).to(
                device=device, dtype=live.dtype
            )
            table_max_abs_error = float((stored - live).abs().max().item())
            if table_max_abs_error >= 1e-2:
                raise ValueError(
                    f"clip {sample.clip_id}: embeddings H5 no corresponden al modelo "
                    f"(max_abs_error={table_max_abs_error:.6f})"
                )
            per_layer_inputs = bridge.get_per_layer_inputs(prompt_ids)
            exact_output = bridge.generate(
                prompt_embeddings(bridge, variant, stored),
                max_new_tokens=max_new_tokens,
                per_layer_inputs=per_layer_inputs,
            )
            exact_rows.append(
                {
                    **row,
                    "prediction": bridge.decode(exact_output)[0].strip(),
                    "token_prediction": prediction,
                    "table_max_abs_error": table_max_abs_error,
                }
            )

        print(
            f"[sweep:{variant.name}] {index}/{len(samples)} "
            f"clip={sample.clip_id}",
            flush=True,
        )

    summary = summarize(rows)
    result = {
        "name": variant.name,
        "pre": variant.pre,
        "post": variant.post,
        "add_bos": variant.add_bos,
        "summary": summary,
        "rows": rows,
    }
    if check_exact_embeddings:
        exact_summary = summarize(exact_rows)
        result["exact_summary"] = exact_summary
        result["exact_chrf_drop"] = summary["chrf"] - exact_summary["chrf"]
        result["exact_rows"] = exact_rows
    return result


def main(
    h5_path: Path,
    split_path: Path,
    out_path: Path,
    model_id: str,
    n_samples: int,
    seed: int,
    max_new_tokens: int,
    exact_top_k: int,
    prompt_set: str,
    fewshot_examples: int,
) -> None:
    samples = load_samples(h5_path, split_path, n_samples, seed)
    if not samples:
        raise ValueError("el split de validación no contiene muestras")

    bridge = GemmaBridge(model_id)
    examples = load_train_examples(h5_path, split_path, fewshot_examples, seed)
    prompts = select_prompts(prompt_set, bridge, examples)
    if not prompts:
        raise ValueError("no hay prompts para ejecutar")

    special_ids = {
        bridge.tokenizer.bos_token_id,
        bridge.tokenizer.eos_token_id,
        bridge.tokenizer.pad_token_id,
    }
    for sample in samples:
        found = special_ids.intersection(int(token) for token in sample.token_ids)
        if found:
            raise ValueError(
                f"clip {sample.clip_id}: special tokens visuales no permitidos: {found}"
            )

    results = []
    for variant in prompts:
        result = run_variant(
            bridge,
            samples,
            variant,
            max_new_tokens=max_new_tokens,
            check_exact_embeddings=False,
        )
        results.append(result)
        payload = {
            "config": {
                "model_id": model_id,
                "n_samples": len(samples),
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "exact_top_k": exact_top_k,
                "prompt_set": prompt_set,
                "fewshot_examples": fewshot_examples,
            },
            "results": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[sweep] {variant.name}: chrF={result['summary']['chrf']:.3f}", flush=True)

    ranked = sorted(results, key=lambda row: row["summary"]["chrf"], reverse=True)
    exact_names = {row["name"] for row in ranked[:exact_top_k]}
    final_results = []
    for variant in prompts:
        if variant.name in exact_names:
            final_results.append(
                run_variant(
                    bridge,
                    samples,
                    variant,
                    max_new_tokens=max_new_tokens,
                    check_exact_embeddings=True,
                )
            )
        else:
            final_results.append(next(row for row in results if row["name"] == variant.name))

    ranked_final = sorted(
        final_results, key=lambda row: row["summary"]["chrf"], reverse=True
    )
    payload = {
        "config": {
            "model_id": model_id,
            "n_samples": len(samples),
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "exact_top_k": exact_top_k,
            "prompt_set": prompt_set,
            "fewshot_examples": fewshot_examples,
        },
        "ranking": [
            {
                "name": row["name"],
                **row["summary"],
                **(
                    {"exact_chrf_drop": row["exact_chrf_drop"]}
                    if "exact_chrf_drop" in row
                    else {}
                ),
            }
            for row in ranked_final
        ],
        "results": final_results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload["ranking"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/processed/dataset2_split_v125.json")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/v125_gemma_oracle_prompt_sweep.json"),
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--exact-top-k", type=int, default=3)
    parser.add_argument(
        "--prompt-set", choices=("base", "strict", "all"), default="base"
    )
    parser.add_argument("--fewshot-examples", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    main(
        args.h5 if args.h5.is_absolute() else root / args.h5,
        args.split if args.split.is_absolute() else root / args.split,
        args.out if args.out.is_absolute() else root / args.out,
        args.model,
        args.n_samples,
        args.seed,
        args.max_new_tokens,
        args.exact_top_k,
        args.prompt_set,
        args.fewshot_examples,
    )
