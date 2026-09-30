import os, json
import torch
from pathlib import Path

class CheckpointManager:
    def __init__(self, base_dir: str, version: int, checkpoint: int):
        self.base = base_dir
        self.v = version

        self.ckpt = checkpoint
        
    def _path(self, extra=""):
        p = os.path.join(self.base, str(self.v), str(self.ckpt), extra)
        os.makedirs(p, exist_ok=True)
        return p

    def load_checkpoint(self, model, optimizer, scheduler):
        base = Path(self.base)
        root = base / str(self.v) / str(self.ckpt - 1)

        epoch_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        if not epoch_dirs:
            raise FileNotFoundError(f"No numeric checkpoint dirs in {root!r}")
        # map dirname → int and find the max
        epochs = [int(p.name) for p in epoch_dirs]
        last = max(epochs)
        model_path = root / str(last) / "checkpoint.pth"
        if not model_path.is_file():
            raise FileNotFoundError(f"Expected checkpoint file at {model_path!r}")

        state = torch.load(model_path, map_location="cuda")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optim_state"])
        if scheduler and state["sched_state"] is not None:
            scheduler.load_state_dict(state["sched_state"])
        return model, optimizer, scheduler

    def save_checkpoint(self, model, epoch, optimizer, scheduler, tag: str | None = None, metadata: dict | None = None):
        subdir = tag if tag is not None else str(epoch)
        path = self._path(subdir)
        raw = getattr(model, "module", model)
        raw = getattr(raw, "_orig_mod", raw)
        state = {
            "epoch": epoch,
            "model_state": raw.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if scheduler else None,
        }
        if metadata is not None:
            state["lineage"] = metadata
        torch.save(state, os.path.join(path, "checkpoint.pth"))

    def save_params(self, params):
        p = self._path()
        with open(os.path.join(p, "parameters.json"), "w") as f:
            json.dump(params, f, indent=2)

    def save_model_architecture(self, model):
        p = self._path()
        with open(os.path.join(p, "model_architecture.txt"), "w") as f:
            print(model, file=f)
