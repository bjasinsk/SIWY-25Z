from loguru import logger
import torch
import typer
import wandb

from siwy.config import DATA_DIR, WANDB_ORG, WANDB_PROJECT

app = typer.Typer()


@app.command()
def main(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to process"),
):
    with wandb.init(project=f"{WANDB_PROJECT}", job_type="example") as run:
        artifact = run.use_artifact(f"{WANDB_ORG}/{WANDB_PROJECT}/{dataset_name}:v0", type="dataset")
        artifact_path = artifact.download(DATA_DIR)

    ds = torch.load(f"{artifact_path}/{dataset_name}.pt", weights_only=False)
    logger.info(f"Loaded dataset {dataset_name} with {len(ds)} samples and classes: {ds.classes}")


if __name__ == "__main__":
    app()

"""
Usage from root directory:

uv run siwy/datasets/use_dataset.py "bus-and-truck-easy-val"
uv run siwy/datasets/use_dataset.py "bus-and-truck-easy-train"
uv run siwy/datasets/use_dataset.py "airplanes"
uv run siwy/datasets/use_dataset.py "dog-and-cat"
"""
