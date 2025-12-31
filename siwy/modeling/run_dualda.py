import pathlib
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
import wandb

from siwy.common import DEVICE, denormalize
from siwy.config import FIGURES_DIR, IS_WINDOWS, MODELS_DIR, PROJ_ROOT
from siwy.datasets.CatDogConfig import CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE, CLASS_TO_IDX
from siwy.datasets.common import DEFAULT_TRANSFORM, load_dataset
from siwy.datasets.wrapper import LabelToIdxWrapper
from siwy.ModelsFactory import construct_rn18

from dualDA.explainers import DualDA

if IS_WINDOWS:
    pathlib.PosixPath = pathlib.WindowsPath

# TODO: fix this file
LOCAL_CKPT_PATH = PROJ_ROOT / "artifacts" / "cat-dog-2025-12-23-17-17-44-model-0-epoch-7-v0"

USE_LOCAL = False


# TODO: move to common utils with plot_trak
def plot_dualda_results(
    run, train_loader, test_loader, dualda_matrix, test_indices, top_k=5, dataset="dog-and-cat"
):
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
        scores = dualda_matrix[:, test_idx]
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
        fig_path = FIGURES_DIR / f"dualda_{dataset}_{test_idx}.png"
        plt.savefig(fig_path)
        run.log({f"dualda_{dataset}_{test_idx}": wandb.Image(fig)})
        plt.close(fig)

    run.log({f"dualda_{dataset}_scores": summary_table})


def main(dataset="dog-and-cat", ood_dataset="airplanes", batch_size=5, num_classes=3, lr=0.001, epochs=None, top_k=5):
    if epochs is None:
        epochs = [8]
    model = construct_rn18(num_classes=num_classes, weights=None).to(DEVICE)
    with wandb.init(project="SIWY-25Z", job_type="dualda") as run:
        run.config.update(
            {
                "model": "resnet18-pretrained",
                "dataset": dataset,
                "ood_dataset": "airplanes",
                "batch_size": batch_size,
                "num_classes": num_classes,
                "epochs": epochs,
                "lr": lr,
                "top_k": top_k,
            }
        )

        artifact_root_dir = MODELS_DIR / dataset
        artifact_root_dir.mkdir(parents=True, exist_ok=True)

        # --- LOAD CHECKPOINTS ---
        if not USE_LOCAL:
            # TODO: improve loading multiple epochs
            for epoch in epochs:
                artifact = run.use_artifact(CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE.format(epoch), type="model")

                artifact_root_dir_epoch = artifact_root_dir / f"epoch_{epoch}"
                if not artifact_root_dir_epoch.exists():  # TODO: fix this check
                    artifact_root_dir_epoch.mkdir(parents=True, exist_ok=True)
                    artifact.download(root=artifact_root_dir_epoch)

            ckpt_files = sorted(list(Path(artifact_root_dir).glob("**/*.pt")))
            logger.debug(f"ckpt_files: {ckpt_files}")
            assert len(ckpt_files) > 0, "No checkpoint found in artifact!"
            # Załaduj pierwszy checkpoint do modelu (np. epoch_1)
            model.load_state_dict(torch.load(ckpt_files[0], map_location=DEVICE))
            print(f"Loaded checkpoint from wandb: {ckpt_files[0]}")
        else:
            model.load_state_dict(torch.load(LOCAL_CKPT_PATH, map_location=DEVICE))
            print(f"Loaded local checkpoint: {LOCAL_CKPT_PATH}")
            weights_paths = [str(LOCAL_CKPT_PATH)]
            model.eval()

        # --- DATA ---
        dog_cat_ds = load_dataset(dataset)
        airplane_ds = load_dataset(ood_dataset)
        logger.debug(f"Airplanes dataset: {airplane_ds}")
        train_ds = dog_cat_ds["train"]
        test_ds = airplane_ds["test"]
        # TODO: get better idicies for airplane dataset
        test_ds = LabelToIdxWrapper(base_ds=test_ds, class_to_idx=CLASS_TO_IDX, transform=DEFAULT_TRANSFORM)

        logger.debug(f"Test dataset size: {len(test_ds)}")

        if hasattr(train_ds.dataset, "transform"):
            train_ds.dataset.transform = DEFAULT_TRANSFORM

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        # --- DUALDA ---

        C = 0.001
        device = "cuda"
        cache_dir = "/content/cache_dir"
        features_dir = "/content/features_dir"

        explainer = DualDA(
            model,
            train_ds,  # Changed from 'train' function to 'train_dataset'
            device=device,
            dir=cache_dir,
            features_dir=features_dir,
            C=C,
        )

        # --- SAVE SCORES TO WANDB ---
        matrix_path = MODELS_DIR / "dualda_score_matrix.pt"
        torch.save(explainer, matrix_path)
        artifact = wandb.Artifact(f"dualda-{dataset}", type="dudalda-scores")
        artifact.add_file(matrix_path)
        run.log_artifact(artifact)

        # --- PLOT RESULTS ---
        plot_dualda_results(
            run, train_loader, test_loader, explainer, test_indices=list(range(top_k)), top_k=top_k
        )

    logger.success("DualDA finished successfully!")


# TODO: move to common utils with plot_trak
def plot_no_train(batch_size, dataset="dog-and-cat", ood_dataset="airplanes", top_k=5):
    with wandb.init(project="SIWY-25Z", job_type="dualda-wyniki") as run:
        dog_cat_ds = load_dataset(dataset)
        airplane_ds = load_dataset(ood_dataset)
        logger.debug(f"Airplanes dataset: {airplane_ds}")
        train_ds = dog_cat_ds["train"]
        test_ds = airplane_ds["test"]
        test_ds = LabelToIdxWrapper(base_ds=test_ds, class_to_idx=CLASS_TO_IDX, transform=DEFAULT_TRANSFORM)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        plot_dualda_results(
            run, train_loader, test_loader, dualda_matrix=torch.load("dualda_score_matrix.pt"), test_idx=0, top_k=top_k
        )

    logger.success("plotting DualDA finished successfully!")


if __name__ == "__main__":
    main()
    # plot_no_train(16)
