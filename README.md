# Multimodal Sign Language Model

Research code for mapping sign-language video keypoints to text-token sequences. The current Imitator prototype predicts Gemma token IDs from keypoint sequences; a separate Gemma stage can correct and format predictions. This is research software, not a production translation system.

## Requirements

- Python 3.11 or newer (the package metadata currently requires 3.11.11+)
- A compatible PyTorch environment; GPU support is recommended for training
- Dataset files and, for Gemma experiments, model/tokenizer files obtained under their own terms

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The broad `requirements.txt` also lists packages used by auxiliary scripts; choose PyTorch and CUDA versions compatible with your hardware before installing that full environment. The project does not distribute datasets, pretrained weights, or model caches. Obtain each resource from its provider, check its terms, and place or pass its path to the relevant script. Training commands require prepared HDF5 data and token embeddings.

## Run

Run commands from the repository root. Use `--help` to see data and output path options for each entrypoint.

```bash
PYTHONPATH=. python scripts/train/train_imitator_dataset1_tokens.py --help
PYTHONPATH=. python scripts/train/train_temporal_v126.py --help
PYTHONPATH=. python scripts/train/train_isolated_staged.py --help
```

Experiment configurations are in [`experiments/`](experiments/). Tests are in [`tests/`](tests/).

## License

Copyright (c) 2026 Christian Velasquez, Giorgio Mancusi, and Rody Vilchez.

Project code and materials are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), unless a file or component states otherwise. Third-party software, datasets, and pretrained models retain their own terms. See [LICENSE](LICENSE).
