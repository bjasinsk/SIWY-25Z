from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch

from siwy.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from siwy.common import plot_multi_method_explainability
from siwy.datasets.common import load_dataset
from siwy.datasets.wrapper import LabelToIdxWrapper
from siwy.datasets.CatDogConfig import CLASS_TO_IDX

app = typer.Typer()


@app.command()
def trak():
    # TODO: implement trak - plot_no_train with optional wandb getting/logging
    pass


@app.command()
def tracin():
    # TODO: implement tracin - plot_no_train with optional wandb getting/logging
    pass


@app.command()
def dualda():
    # TODO: implement dualda functionality
    pass


def get_one_method_scores(run, method_name, dataset, artifact_type=None, is_torch=True):
    WANDB_PATH = f"jarcin/SIWY-25Z/{method_name}-{dataset}:latest"
    SCORES_DIR = REPORTS_DIR / "scores"

    if artifact_type is None:
        artifact_type = f"{method_name}-scores"
    artifact = run.use_artifact(WANDB_PATH, type=artifact_type)

    if is_torch:
        local_artifact_path = artifact.get_entry(f"{method_name}_score_matrix.pt").download(root=SCORES_DIR)
        result = torch.load(local_artifact_path).detach().cpu().numpy()
    else:
        local_artifact_path = artifact.get_entry("quickstart.mmap").download(root=SCORES_DIR)
        result = np.load(local_artifact_path, mmap_mode="r")

    return result


@app.command()
def plot_all_compare(
    dataset: str = typer.Argument(..., help="Name of the dataset to plot"),
    ood_dataset: str = typer.Argument(..., help="Name of the out of ditribution dataset for plotting"),
):
    base_ds = load_dataset(dataset)
    ood_ds = load_dataset(ood_dataset)
    train_ds = base_ds["train"]
    test_ds = ood_ds["test"]
    test_ds = LabelToIdxWrapper(base_ds=test_ds, class_to_idx=CLASS_TO_IDX)

    with wandb.init(project="SIWY-25Z", job_type="plot") as run:
        dualda_result = get_one_method_scores(run, "dualda", dataset, artifact_type="dudalda-scores")
        trak_result = get_one_method_scores(run, "trak", dataset, is_torch=False)
        tracin_result = get_one_method_scores(run, "tracin", dataset)

        all_scores = np.array([dualda_result, trak_result, tracin_result])
        plot_multi_method_explainability(
            run, train_ds, test_ds, all_scores, method_names=["DualDA", "TRAK", "TracIn"], dataset_name=dataset
        )


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()


"""
Usage from root directory:
# dog-and-cat
uv run siwy/plots.py plot-all-compare "dog-and-cat" "airplanes"
"""
