# Experiment configurations

Versioned TOML configurations live here; shared defaults are in `config/`. Run training entrypoints from the repository root and use `--help` for supported options. For example:

```bash
MSLM_EXPERIMENT_CONFIG=experiments/v119_ctc/ctc_v119.toml PYTHONPATH=. python scripts/train/train_ctc_v119.py
```

Configurations describe experiments; they do not include datasets, checkpoints, or pretrained models.
