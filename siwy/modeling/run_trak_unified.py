"""
Unified TRAK evaluation script supporting multiple datasets.
Consolidates run_trak.py, run_trak_bus_truck.py, and run_trak_horse_elephant.py.
"""

from datetime import datetime

from loguru import logger
from matplotlib import pyplot as plt
from numpy.lib.format import open_memmap
import torch
from torch import Tensor
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from trak import TRAKer
import typer
import wandb

from siwy.common import DEVICE, denormalize
from siwy.config import FIGURES_DIR, MODELS_DIR
from siwy.datasets.common import DEFAULT_TRANSFORM, load_dataset
from siwy.datasets.wrapper import LabelToIdxWrapper
from siwy.modeling.common_configs import DATASET_CONFIGS
from siwy.modeling.modeling_utils import (
    get_train_dataset,
    load_checkpoints_from_wandb,
    setup_windows_compatibility,
)
from siwy.ModelsFactory import construct_rn18

# Apply Windows compatibility fixes
setup_windows_compatibility()

app = typer.Typer()


def plot_trak(run, ds_train: ImageFolder, ds_val: ImageFolder, scores: Tensor, top_k: int, dataset: str):
    """
    Plot TRAK top contributors for each validation image.

    Args:
        run: wandb run object
        ds_train: Training dataset
        ds_val: Validation/test dataset
        scores: TRAK scores matrix (train_size x val_size)
        top_k: Number of top contributors to visualize
        dataset: Dataset name for labeling
    """
    summary_table = wandb.Table(columns=["test_id", "train_id", "score"])

    for i in range(len(ds_val)):
        fig, axs = plt.subplots(ncols=7, figsize=(15, 3))
        fig.suptitle("Top scoring TRAK images from the train set")

        # Show target image
        axs[0].imshow(denormalize(ds_val[i][0].permute(1, 2, 0)).clamp(0, 1))
        axs[0].axis("off")
        axs[0].set_title("Target image")
        axs[1].axis("off")

        logger.info(f"val class {ds_val[i][1]}")
        top_trak_scorers = scores[:, i].argsort()[-top_k:][::-1]

        # Convert to lists for wandb table
        if isinstance(scores, torch.Tensor):
            trak_scorers_list = top_trak_scorers.cpu().tolist()
            scores_list = scores[top_trak_scorers].cpu().tolist()
        else:
            # Assuming numpy
            trak_scorers_list = top_trak_scorers.tolist()
            scores_list = scores[top_trak_scorers].tolist()

        summary_table.add_data(i, trak_scorers_list, scores_list)

        logger.info(f"Test idx: {i}, top indices: {top_trak_scorers}, scores: {scores[top_trak_scorers]}")

        # Show top contributing training images
        for ii, train_im_ind in enumerate(top_trak_scorers):
            logger.info(f"train id ({train_im_ind}): {ds_train[train_im_ind][1]}")
            axs[ii + 2].imshow(denormalize(ds_train[train_im_ind][0].permute(1, 2, 0)))
            axs[ii + 2].axis("off")

        logger.info("=" * 40)
        fig.show()
        plt.savefig(FIGURES_DIR / f"trak_{dataset}_val_image_{i}.png")
        run.log({"trak_results": wandb.Image(fig)})

    run.log({f"trak_{dataset}_scores": summary_table})


@app.command()
def main(
    dataset: str = typer.Option(
        "dog-and-cat", help="Dataset: dog-and-cat | bus-and-truck-easy-train | horse-and-elephant-easy-train"
    ),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    epochs: list[int] = typer.Option(None, help="List of epochs to evaluate (uses config defaults if not provided)"),
    top_k: int = typer.Option(5, help="Number of top contributors to plot"),
    ood_dataset: str = typer.Option("airplanes", help="Out-of-distribution dataset for testing"),
):
    """
    Run TRAK evaluation on specified dataset.
    """
    # Load dataset-specific configuration
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset]

    # Use default epochs if not specified
    if epochs is None:
        epochs = config["default_epochs"]

    num_classes = config["num_classes"]

    # Setup paths
    datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_path = MODELS_DIR / "trak_results" / datetime_str
    results_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = (
        construct_rn18(num_classes=num_classes, weights=None).to(memory_format=torch.channels_last).to(DEVICE).eval()
    )

    # Initialize wandb
    with wandb.init(project="SIWY-25Z", job_type="trak") as run:
        run.config.update(
            {
                "model": "resnet18-pretrained",
                "dataset": dataset,
                "batch_size": batch_size,
                "num_classes": num_classes,
                "ood_dataset": ood_dataset,
                "epochs": epochs,
                "top_k": top_k,
            }
        )

        # Load checkpoints
        logger.info(f"Loading checkpoints for epochs: {epochs}")
        ckpts = load_checkpoints_from_wandb(
            run=run,
            artifact_template=config["artifact_template"],
            epochs=epochs,
            models_dir=MODELS_DIR,
            dataset=dataset,
        )

        # Load datasets
        logger.info(f"Loading dataset: {dataset}")
        train_data = load_dataset(dataset)
        ood_data = load_dataset(ood_dataset)

        # Handle dataset split strategy
        train_ds = get_train_dataset(train_data, config["train_split"])
        test_ds = ood_data["test"]
        test_ds = LabelToIdxWrapper(base_ds=test_ds, class_to_idx=config["class_to_idx"], transform=DEFAULT_TRANSFORM)

        # Ensure consistent transforms
        if hasattr(train_ds, "dataset") and hasattr(train_ds.dataset, "transform"):
            train_ds.dataset.transform = DEFAULT_TRANSFORM

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

        logger.info("Starting TRAK evaluation...")
        logger.info(f"Train set size: {len(train_ds)}")
        logger.info(f"Test set size: {len(test_ds)}")

        # Create results directory
        (results_path / "scores").mkdir(parents=True, exist_ok=True)

        # Initialize TRAKer
        traker = TRAKer(
            model=model,
            task="image_classification",
            proj_dim=4096,
            train_set_size=len(train_ds),
            device=DEVICE,
            use_half_precision=True,
            save_dir=results_path,
        )

        # Featurize training set with all checkpoints
        logger.info(f"Featurizing training set with {len(ckpts)} checkpoints...")
        for model_id, ckpt in enumerate(tqdm(ckpts, desc="Checkpoints")):
            traker.load_checkpoint(ckpt, model_id=model_id)
            for batch in tqdm(train_loader, desc=f"Featurize epoch {epochs[model_id]}", leave=False):
                batch = [x.to(DEVICE) for x in batch]
                traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        traker.finalize_features()

        # Score test set with all checkpoints
        logger.info("Scoring test set...")
        for model_id, ckpt in enumerate(tqdm(ckpts, desc="Scoring")):
            traker.start_scoring_checkpoint(
                exp_name="quickstart", checkpoint=ckpt, model_id=model_id, num_targets=len(test_loader.dataset)
            )
            for batch in test_loader:
                batch = [x.to(DEVICE) for x in batch]
                traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name="quickstart")
        _scores = open_memmap(results_path / "scores" / "quickstart.mmap")

        # Save scores to wandb
        scores_artifact = wandb.Artifact(
            name=f"trak-{dataset}",
            type="trak-scores",
        )
        scores_artifact.add_file(results_path / "scores" / "quickstart.mmap")
        run.log_artifact(scores_artifact)

        # Plot results
        logger.info("Generating visualizations...")
        plot_trak(run, train_ds, test_ds, scores, top_k, dataset=dataset)

    logger.success("TRAK evaluation finished successfully!")


if __name__ == "__main__":
    app()
