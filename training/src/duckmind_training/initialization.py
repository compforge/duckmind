"""Adapt only the pretrained state projection when body feedback exceeds its padding."""

import json
import logging
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def expanded_state_checkpoint(
    model: str, revision: str | None, output: Path, state_dim: int
) -> Path:
    """Materialize a native checkpoint; preserve every weight except appended zero columns."""
    source = (
        Path(model)
        if Path(model).is_dir()
        else Path(
            snapshot_download(model, revision=revision, allow_patterns=["*.json", "*.safetensors"])
        )
    )
    config = json.loads((source / "config.json").read_text())
    if state_dim <= config["max_state_dim"]:
        return source
    shutil.copytree(source, output)
    weights = load_file(str(source / "model.safetensors"))
    key = "model.state_proj.weight"
    old = weights[key]
    if old.shape[1] != config["max_state_dim"]:
        raise ValueError("pretrained state projection and configuration disagree")
    # Zero new columns preserves the pretrained function for zero-padded observations.
    weights[key] = torch.nn.functional.pad(old, (0, state_dim - old.shape[1]))
    save_file(weights, str(output / "model.safetensors"), metadata={"format": "pt"})
    config["max_state_dim"] = state_dim
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "state-expansion.json").write_text(
        json.dumps(
            {
                "source": model,
                "revision": revision,
                "old_dim": old.shape[1],
                "new_dim": state_dim,
            },
            indent=2,
        )
    )
    logging.info("expanded pretrained state projection from %s to %s", old.shape[1], state_dim)
    return output
