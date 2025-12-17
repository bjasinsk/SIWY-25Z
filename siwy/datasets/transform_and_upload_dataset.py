from pathlib import Path

from loguru import logger
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import typer
import wandb

from siwy.config import PROCESSED_DATA_DIR, WANDB_PROJECT, SEED
from siwy.datasets.AirplaneDataset import AirplaneDataset as AirplaneDatasetClass
from torch.utils.data import random_split

GENERATOR = torch.manual_seed(SEED)
TORCH_DATASETS = {
    "ImageFolder": ImageFolder,
    "Airplane": AirplaneDatasetClass,
}

DATASETS = ["bus-and-truck-easy-val", "bus-and-truck-easy-train", "airplanes", "dog-and-cat"]

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

app = typer.Typer()


@app.command()
def main(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to process"),
    dataset_dir: Path = typer.Argument(..., help="Path to the dataset directory"),
    cls: str = typer.Option(default="ImageFolder", help="Class of the dataset to process"),
    upload: bool = typer.Option(default=False, help="Upload the processed dataset to WB"),
    overwrite: bool = typer.Option(default=False, help="Overwrite existing processed dataset"),
):
    if not dataset_dir.is_dir():
        logger.error(f"Dataset directory {dataset_dir} does not exist.")
        raise typer.Exit(code=1)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory {dataset_dir} does not exist.")
        raise typer.Exit(code=1)

    if (PROCESSED_DATA_DIR / f"{dataset_name}.pt").exists() and not overwrite:
        logger.info(f"Processed dataset {dataset_name} already exists and overwrite is disabled. Skipping processing.")
        raise typer.Exit(code=0)

    logger.info(f"Processing {dataset_name} at {dataset_dir}")
    ds = TORCH_DATASETS.get(cls, None)(
        root=dataset_dir,
        transform=DEFAULT_TRANSFORM,
    )
    train_ds, val_ds, test_ds = random_split(ds, [0.7, 0.2, 0.1], generator=GENERATOR)

    logger.info(f"Classes: {ds.classes}")
    logger.info(f"Dataset size: {len(ds)}")

    processed_path = PROCESSED_DATA_DIR / f"{dataset_name}.pt"
    logger.info(f"Saving processed dataset to {processed_path}")

    torch.save({
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "classes": ds.classes,
    }, processed_path)

    if not upload:
        logger.info(f"Saving processed dataset to {PROCESSED_DATA_DIR}")
        raise typer.Exit(code=0)

    logger.info(f"Uploading dataset {dataset_name} to WB")
    with wandb.init(project=WANDB_PROJECT, job_type="add-dataset") as run:
        artifact = wandb.Artifact(name=dataset_name, type="dataset")
        artifact.add_file(local_path=processed_path, name=f"{dataset_name}.pt", policy="immutable")
        run.log_artifact(artifact)

    logger.info(f"Uploaded dataset {dataset_name} to WB")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()

"""
Usage from root directory:

uv run siwy/datasets/dataset.py "bus-and-truck-easy-val" "data/raw/task2/easy/val" --overwrite --upload
uv run siwy/datasets/dataset.py "bus-and-truck-easy-train" "data/raw/task2/easy/train" --overwrite --upload
uv run siwy/datasets/dataset.py "airplanes" "data/raw/1_Liner TF" --overwrite --cls Airplane --upload
uv run siwy/datasets/dataset.py "dog-and-cat" "data/raw/PetImages" --overwrite --upload --cls DogAndCat
"""
