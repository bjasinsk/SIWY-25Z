import pathlib

from loguru import logger
import torch
import typer

from siwy.config import PROCESSED_DATA_DIR, WANDB_DATASET_PATH, WANDB_PROJECT
import wandb

app = typer.Typer()

pathlib.PosixPath = pathlib.WindowsPath


@app.command()
def main(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to process"),
):
    with wandb.init(project=f"{WANDB_PROJECT}", job_type="example") as run:
        artifact = run.use_artifact(WANDB_DATASET_PATH(dataset_name), type="dataset")
        artifact_path = artifact.download(PROCESSED_DATA_DIR)

    ds = torch.load(f"{artifact_path}/{dataset_name}.pt", weights_only=False)

    # Handle both dictionary format (with train/val/test splits) and raw dataset format
    if isinstance(ds, dict):
        # Dictionary format with splits
        if "val" not in ds or "test" not in ds or "train" not in ds:
            logger.error(f"Dataset dictionary missing required keys. Available keys: {ds.keys()}")
            raise KeyError(f"Dataset dictionary missing required keys. Available keys: {ds.keys()}")

        logger.info(f"Train samples: {len(ds['train'])}")
        logger.info(f"Val samples: {len(ds['val'])}")
        logger.info(f"Test samples: {len(ds['test'])}")
        logger.info(
            f"Loaded dataset {dataset_name} with {len(ds['train'])} train samples, {len(ds['val'])} val samples, {len(ds['test'])} test samples and classes: {ds['classes']}"
        )
    else:
        # Raw dataset format (ImageFolder, etc.)
        logger.info(f"Dataset type: {type(ds).__name__}")
        logger.info(f"Total samples: {len(ds)}")
        if hasattr(ds, "classes"):
            logger.info(f"Classes: {ds.classes}")
        if hasattr(ds, "class_to_idx"):
            logger.info(f"Number of classes: {len(ds.class_to_idx)}")
        logger.info(f"Loaded dataset {dataset_name} with {len(ds)} samples")


if __name__ == "__main__":
    app()
    # main("dog-and-cat")
    # main("airplanes")

"""
Usage from root directory:

uv run siwy/datasets/use_dataset.py "bus-and-truck-easy-val"
uv run siwy/datasets/use_dataset.py "bus-and-truck-easy-train"
uv run siwy/datasets/use_dataset.py "airplanes"
uv run siwy/datasets/use_dataset.py "dog-and-cat"
"""
