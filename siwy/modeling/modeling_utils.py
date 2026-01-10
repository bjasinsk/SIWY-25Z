"""
Shared utilities for model evaluation scripts (TRAK, TracIn).
"""

import os
import pathlib
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from trak.savers import MmapSaver

from siwy.common import DEVICE
from siwy.config import IS_WINDOWS


def setup_windows_compatibility():
    """
    Apply Windows-specific compatibility fixes for TRAK's MmapSaver.
    Monkeypatches MmapSaver.init_store to explicitly release file handles.
    """

    def patched_init_store(self, model_id) -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        if os.path.exists(prefix):
            self.logger.info(f"Model ID folder {prefix} already exists")
        os.makedirs(prefix, exist_ok=True)
        featurized_so_far = np.zeros(shape=(self.train_set_size,), dtype=np.int32)
        ft = self._load(
            prefix.joinpath("_is_featurized.mmap"),
            shape=(self.train_set_size,),
            mode="w+",
            dtype=np.int32,
        )
        if ft is not None:
            ft[:] = featurized_so_far[:]
            ft.flush()
            # Explicitly release the file handle
            del ft

        self.load_current_store(model_id, mode="w+")

    if IS_WINDOWS:
        pathlib.PosixPath = pathlib.WindowsPath
        MmapSaver.init_store = patched_init_store


def load_checkpoints_from_wandb(run, artifact_template, epochs, models_dir, dataset):
    """
    Load multiple epoch checkpoints from wandb artifacts.

    Args:
        run: wandb run object
        artifact_template: Template string with {} placeholder for epoch number
        epochs: List of epoch numbers to load
        models_dir: Base directory for storing model artifacts
        dataset: Dataset name for organizing checkpoints

    Returns:
        List of loaded checkpoint dictionaries
    """
    artifact_root_dir = models_dir / dataset
    artifact_root_dir.mkdir(parents=True, exist_ok=True)

    # Download artifacts for each epoch
    for epoch in epochs:
        artifact = run.use_artifact(artifact_template.format(epoch), type="model")

        artifact_root_dir_epoch = artifact_root_dir / f"epoch_{epoch}"
        if not artifact_root_dir_epoch.exists():
            artifact_root_dir_epoch.mkdir(parents=True, exist_ok=True)
            artifact.download(root=artifact_root_dir_epoch)

    # Load all checkpoints
    ckpt_files = sorted(list(Path(artifact_root_dir).glob("**/*.pt")))
    logger.debug(f"ckpt_files: {ckpt_files}")
    assert len(ckpt_files) > 0, "No checkpoint found in artifact!"

    ckpts = [torch.load(ckpt, map_location=DEVICE) for ckpt in ckpt_files]
    logger.debug(f"Loaded {len(ckpts)} checkpoints for evaluation.")

    return ckpts


def get_train_dataset(loaded_ds, split_key):
    """
    Handle dataset access for different split strategies.

    Args:
        loaded_ds: Loaded dataset (may have splits or be a single dataset)
        split_key: Key to access split (e.g., "train") or None for direct access

    Returns:
        Training dataset
    """
    return loaded_ds[split_key] if split_key else loaded_ds
