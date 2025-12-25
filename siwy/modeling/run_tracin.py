import pathlib
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss

# TODO: import tracin from uv
from TracInPyTorch.src.tracin import vectorized_calculate_tracin_score
import wandb

from siwy.common import DEVICE, denormalize
from siwy.config import FIGURES_DIR, IS_WINDOWS, MODELS_DIR, PROJ_ROOT
from siwy.datasets.CatDogConfig import CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE, CLASS_TO_IDX
from siwy.datasets.common import DEFAULT_TRANSFORM, load_dataset
from siwy.datasets.wrapper import LabelToIdxWrapper
from siwy.ModelsFactory import construct_rn18

if IS_WINDOWS:
    pathlib.PosixPath = pathlib.WindowsPath

# TODO: fix this file
LOCAL_CKPT_PATH = PROJ_ROOT / "artifacts" / "cat-dog-2025-12-23-17-17-44-model-0-epoch-7-v0"

USE_LOCAL = False


# TODO: move to common utils with plot_trak
def plot_tracin_top_contributors(run, train_loader, test_loader, tracin_matrix, test_indices, top_k=5):
    train_imgs = []
    for batch in train_loader:
        ims, _ = batch
        train_imgs.extend([im for im in ims])
    test_imgs = []
    for batch in test_loader:
        ims, _ = batch
        test_imgs.extend([im for im in ims])

    for test_idx in test_indices:
        scores = tracin_matrix[:, test_idx]
        top_indices = torch.argsort(scores, descending=True)[:top_k]

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
        fig_path = FIGURES_DIR / f"tracin_dogcat_{test_idx}.png"
        plt.savefig(fig_path)
        run.log({f"tracin_dogcat_{test_idx}": wandb.Image(fig)})
        plt.close(fig)


def main(dataset="dog-and-cat", ood_dataset="airplanes", batch_size=5, num_classes=3, lr=0.001, epochs=None, top_k=5):
    if epochs is None:
        epochs = [0, 1, 2, 4, 6, 8]
    model = construct_rn18(num_classes=num_classes, weights=None).to(DEVICE)
    with wandb.init(project="SIWY-25Z", job_type="tracin") as run:
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

            ckpt_files = list(Path(artifact_root_dir).glob("**/*.pt"))
            logger.debug(f"ckpt_files: {ckpt_files}")
            assert len(ckpt_files) > 0, "No checkpoint found in artifact!"
            # Załaduj pierwszy checkpoint do modelu (np. epoch_1)
            # model.load_state_dict(torch.load(ckpt_files[0], map_location=DEVICE))
            # print(f"Loaded checkpoint from wandb: {ckpt_files[0]}")
            weights_paths = ckpt_files
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

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        # --- TRACIN ---
        criterion = CrossEntropyLoss(label_smoothing=0.0, reduction="none")

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
        matrix_path = MODELS_DIR / "tracin_score_matrix.pt"
        torch.save(matrix, matrix_path)
        artifact = wandb.Artifact(f"tracin-matrix-{dataset}", type="result")
        artifact.add_file(matrix_path)
        run.log_artifact(artifact)

        # --- PLOT RESULTS ---
        plot_tracin_top_contributors(
            run, train_loader, test_loader, matrix, test_indices=list(range(top_k)), top_k=top_k
        )

    logger.success("Tracin finished successfully!")


# TODO: move to common utils with plot_trak
def plot_no_train(batch_size, dataset="dog-and-cat", ood_dataset="airplanes", top_k=5):
    with wandb.init(project="SIWY-25Z", job_type="tracin-wyniki") as run:
        dog_cat_ds = load_dataset(dataset)
        airplane_ds = load_dataset(ood_dataset)
        logger.debug(f"Airplanes dataset: {airplane_ds}")
        train_ds = dog_cat_ds["train"]
        test_ds = airplane_ds["test"]
        test_ds = LabelToIdxWrapper(base_ds=test_ds, class_to_idx=CLASS_TO_IDX, transform=DEFAULT_TRANSFORM)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        plot_tracin_top_contributors(
            run, train_loader, test_loader, tracin_matrix=torch.load("tracin_score_matrix.pt"), test_idx=0, top_k=top_k
        )

    logger.success("plotting Tracin finished successfully!")


if __name__ == "__main__":
    main()
    # plot_no_train(16)
