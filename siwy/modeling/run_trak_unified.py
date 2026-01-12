"""
Unified TRAK evaluation script supporting multiple datasets.
"""

from datetime import datetime

from loguru import logger
from numpy.lib.format import open_memmap
import torch
from tqdm import tqdm
from trak import TRAKer
import typer
import wandb

from siwy.common import DEVICE, plot_explainability_results
from siwy.config import MODELS_DIR
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

# app = typer.Typer()


# @app.command()
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
        elif hasattr(train_ds, "transform"):
            train_ds.transform = DEFAULT_TRANSFORM

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
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        plot_explainability_results(run, train_ds, test_ds, scores, logger, "TRAK", top_k, dataset_name=dataset)

    logger.success("TRAK evaluation finished successfully!")


if __name__ == "__main__":
    main(dataset="bus-and-truck-easy-train", epochs=None, ood_dataset="airplanes", batch_size=16, top_k=5)
