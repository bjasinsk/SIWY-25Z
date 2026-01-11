"""
Unified TracIn script for all datasets.
Consolidates run_tracin.py and run_tracin_bus_truck.py.
"""

import pathlib
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from tracin_pytorch.tracin import vectorized_calculate_tracin_score
import typer
import wandb

from siwy.common import DEVICE, denormalize
from siwy.config import FIGURES_DIR, IS_WINDOWS, MODELS_DIR
from siwy.datasets.common import DEFAULT_TRANSFORM, load_dataset
from siwy.datasets.wrapper import LabelToIdxWrapper
from siwy.modeling.common_configs import DATASET_CONFIGS
from siwy.modeling.modeling_utils import get_train_dataset
from siwy.ModelsFactory import construct_rn18

if IS_WINDOWS:
    pathlib.PosixPath = pathlib.WindowsPath


def plot_tracin_top_contributors(
    run, train_loader, test_loader, tracin_matrix, test_indices, top_k=5, dataset="dog-and-cat"
):
    """Plot top contributing training samples for each test sample."""
    train_imgs = []
    for batch in train_loader:
        ims, _ = batch
        train_imgs.extend([im for im in ims])
    test_imgs = []
    for batch in test_loader:
        ims, _ = batch
        test_imgs.extend([im for im in ims])

    summary_table = wandb.Table(columns=["test_id", "train_id", "score"])

    for test_idx in test_indices:
        scores = tracin_matrix[:, test_idx]
        top_indices = torch.argsort(scores, descending=True)[:top_k]
        summary_table.add_data(test_idx, top_indices.cpu().tolist(), scores[top_indices].cpu().tolist())

        logger.info(f"Test idx: {test_idx}, top indices: {top_indices}, scores: {scores[top_indices]}")
        fig, axs = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
        test_img = denormalize(test_imgs[test_idx].cpu()).clamp(0, 1)
        axs[0].imshow(test_img.permute(1, 2, 0).numpy())
        axs[0].set_title("Test sample")
        axs[0].axis("off")
        for i, idx in enumerate(top_indices):
            train_img = denormalize(train_imgs[idx].cpu()).clamp(0, 1)
            axs[i + 1].imshow(train_img.permute(1, 2, 0).numpy())
            axs[i + 1].set_title(f"Top {i + 1}")
            axs[i + 1].axis("off")
        plt.tight_layout()
        fig_path = FIGURES_DIR / f"tracin_{dataset}_{test_idx}.png"
        plt.savefig(fig_path)
        run.log({f"tracin_{dataset}_{test_idx}": wandb.Image(fig)})
        plt.close(fig)

    run.log({f"tracin_{dataset}_scores": summary_table})


def main(
    dataset: str = typer.Option(
        "dog-and-cat", help="Dataset: dog-and-cat | bus-and-truck-easy-train | horse-and-elephant-easy-train"
    ),
    ood_dataset: str = typer.Option("airplanes", help="Out-of-distribution dataset for testing"),
    batch_size: int = typer.Option(5, help="Batch size for data loading"),
    epochs: list[int] = typer.Option(None, help="Epochs to evaluate (uses dataset defaults if not specified)"),
    top_k: int = typer.Option(5, help="Number of top contributors to visualize"),
    lr: float = typer.Option(0.001, help="Learning rate used during training (for TracIn calculation)"),
    use_local: bool = typer.Option(False, help="Use local checkpoint instead of wandb artifacts"),
    local_ckpt_path: str = typer.Option(None, help="Path to local checkpoint (required if use_local=True)"),
):
    """
    Run TracIn analysis to find influential training samples.

    Supports multiple datasets with automatic configuration lookup.
    """
    # Get dataset configuration
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset]
    if epochs is None:
        epochs = config["default_epochs"]

    num_classes = config["num_classes"]
    artifact_template = config["artifact_template"]
    class_to_idx = config["class_to_idx"]
    train_split = config["train_split"]

    # Handle local checkpoint path
    if use_local and local_ckpt_path is None:
        raise ValueError("local_ckpt_path must be provided when use_local=True")

    if use_local and local_ckpt_path is not None:
        local_ckpt_path = Path(local_ckpt_path)

    # Initialize model
    model = construct_rn18(num_classes=num_classes, weights=None).to(DEVICE)

    with wandb.init(project="SIWY-25Z", job_type="tracin") as run:
        run.config.update(
            {
                "model": "resnet18-pretrained",
                "dataset": dataset,
                "ood_dataset": ood_dataset,
                "batch_size": batch_size,
                "num_classes": num_classes,
                "epochs": epochs,
                "lr": lr,
                "top_k": top_k,
                "use_local": use_local,
            }
        )

        # --- LOAD CHECKPOINTS ---
        if not use_local:
            artifact_root_dir = MODELS_DIR / dataset
            artifact_root_dir.mkdir(parents=True, exist_ok=True)

            # Download artifacts for each epoch
            for epoch in epochs:
                artifact = run.use_artifact(artifact_template.format(epoch), type="model")
                artifact_root_dir_epoch = artifact_root_dir / f"epoch_{epoch}"
                if not artifact_root_dir_epoch.exists():
                    artifact_root_dir_epoch.mkdir(parents=True, exist_ok=True)
                    artifact.download(root=artifact_root_dir_epoch)

            # Get checkpoint file paths
            ckpt_files = sorted(list(Path(artifact_root_dir).glob("**/*.pt")))
            logger.debug(f"ckpt_files: {ckpt_files}")
            assert len(ckpt_files) > 0, "No checkpoint found in artifact!"
            weights_paths = ckpt_files
        else:
            # Use local checkpoint
            model.load_state_dict(torch.load(local_ckpt_path, map_location=DEVICE))
            logger.info(f"Loaded local checkpoint: {local_ckpt_path}")
            weights_paths = [str(local_ckpt_path)]
            model.eval()

        # --- DATA ---
        loaded_ds = load_dataset(dataset)
        airplane_ds = load_dataset(ood_dataset)
        logger.debug(f"OOD dataset ({ood_dataset}): {airplane_ds}")

        # Handle different train split strategies
        train_ds = get_train_dataset(loaded_ds, train_split)
        test_ds = airplane_ds["test"]

        # Wrap test dataset with label mapping
        test_ds = LabelToIdxWrapper(base_ds=test_ds, class_to_idx=class_to_idx, transform=DEFAULT_TRANSFORM)
        logger.debug(f"Test dataset size: {len(test_ds)}")

        # Set transform for training data
        if hasattr(train_ds, "dataset") and hasattr(train_ds.dataset, "transform"):
            train_ds.dataset.transform = DEFAULT_TRANSFORM

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        # --- TRACIN ---
        criterion = CrossEntropyLoss(label_smoothing=0.0, reduction="none")

        logger.info("Computing TracIn scores...")
        matrix = vectorized_calculate_tracin_score(
            model=model,
            criterion=criterion,
            weights_paths=weights_paths,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            lr=lr,
            device=DEVICE,
            use_nested_loop_for_dot_product=False,
            float_labels=False,
        )

        # --- SAVE SCORES TO WANDB ---
        matrix_path = MODELS_DIR / f"tracin_score_matrix_{dataset}.pt"
        torch.save(matrix, matrix_path)
        artifact = wandb.Artifact(f"tracin-{dataset}", type="tracin-scores")
        artifact.add_file(matrix_path)
        run.log_artifact(artifact)
        logger.info(f"Saved TracIn scores to {matrix_path}")

        # --- PLOT RESULTS ---
        plot_tracin_top_contributors(
            run,
            train_loader,
            test_loader,
            matrix,
            test_indices=list(range(top_k)),
            top_k=top_k,
            dataset=dataset,
        )

    logger.success("TracIn finished successfully!")


if __name__ == "__main__":
    typer.run(main)
