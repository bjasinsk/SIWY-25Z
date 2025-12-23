import pathlib
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss

# TODO: import tracin from uv
from TracInPyTorch.src.tracin import vectorized_calculate_tracin_score
from wrapper import LabelToIdxWrapper

from siwy.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from siwy.datasets.transform_and_upload_dataset import DEFAULT_TRANSFORM
from siwy.ModelsFactory import construct_rn18
import wandb

# TODO: use platform check
pathlib.PosixPath = pathlib.WindowsPath

DOG_AND_CAT_MODEL_ARTIFACT_TEMPLATE = "jarcin/SIWY-25Z/cat-dog-2025-12-23-17-17-44-model-0-epoch-{}:v0"
# TODO: fix this file
LOCAL_CKPT_PATH = PROJ_ROOT / "artifacts" / "cat-dog-2025-12-23-17-17-44-model-0-epoch-7-v0"

AIRPLANES_DATASET_PATH = PROCESSED_DATA_DIR / "airplanes.pt"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
USE_LOCAL = False


def denormalize(img, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return img * std + mean


def plot_tracin_top_contributors(run, train_loader, test_loader, tracin_matrix, test_idx=0, top_k=5):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_imgs = []
    for batch in train_loader:
        ims, _ = batch
        train_imgs.extend([im for im in ims])
    test_imgs = []
    for batch in test_loader:
        ims, _ = batch
        test_imgs.extend([im for im in ims])
    scores = tracin_matrix[:, test_idx]
    logger.debug(f"scores: {scores}")
    logger.debug(f"type(scores): {type(scores)}")

    if isinstance(scores, torch.Tensor):
        top_indices = torch.argsort(scores, descending=True)[:top_k]
        logger.debug("tensorem jestem")
    else:
        import numpy as np

        logger.debug("tensorem nie jestem")
        scores = np.array(scores)
        top_indices = scores.argsort()[::-1][:top_k]

    fig, axs = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
    test_img = denormalize(test_imgs[test_idx].cpu(), mean, std).clamp(0, 1)
    axs[0].imshow(test_img.permute(1, 2, 0).numpy())
    axs[0].set_title("Test sample")
    axs[0].axis("off")
    for i, idx in enumerate(top_indices):
        train_img = denormalize(train_imgs[idx].cpu(), mean, std).clamp(0, 1)
        axs[i + 1].imshow(train_img.permute(1, 2, 0).numpy())
        axs[i + 1].set_title(f"Top {i + 1}")
        axs[i + 1].axis("off")
    plt.tight_layout()
    fig_path = f"tracin_top_{test_idx}.png"
    plt.savefig(fig_path)
    run.log({f"tracin_top_{test_idx}": wandb.Image(fig)})
    plt.close(fig)


def plot_tracin_top_contributors_2(run, train_loader, test_loader, tracin_matrix, test_idx=0, top_k=5):
    # Pobierz obrazy z loaderów
    train_imgs = []
    for batch in train_loader:
        ims, _ = batch
        train_imgs.extend([im for im in ims])
    test_imgs = []
    for batch in test_loader:
        ims, _ = batch
        test_imgs.extend([im for im in ims])
    # scores = tracin_matrix[:, test_idx]
    # top_indices = scores.argsort()[::-1][:top_k]
    scores = tracin_matrix[:, test_idx]
    logger.debug(f"scores: {scores}")
    logger.debug(f"type(scores): {type(scores)}")

    if isinstance(scores, torch.Tensor):
        top_indices = torch.argsort(scores, descending=True)[:top_k]
        logger.debug("tensorem jestem")
        logger.debug("dupa")
    else:
        import numpy as np

        logger.debug("tensorem nie jestem")
        scores = np.array(scores)
        top_indices = scores.argsort()[::-1][:top_k]

    fig, axs = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
    test_img = test_imgs[test_idx]
    axs[0].imshow(test_img.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Test sample")
    axs[0].axis("off")
    for i, idx in enumerate(top_indices):
        train_img = train_imgs[idx]
        axs[i + 1].imshow(train_img.permute(1, 2, 0).cpu().numpy())
        axs[i + 1].set_title(f"Top {i + 1}")
        axs[i + 1].axis("off")
    plt.tight_layout()
    fig_path = f"tracin_top_{test_idx}.png"
    plt.savefig(fig_path)
    run.log({f"tracin_top_{test_idx}": wandb.Image(fig)})
    plt.close(fig)


def main(dataset="dog-and-cat"):
    model = construct_rn18(num_classes=2, weights=None).to(DEVICE)
    run = wandb.init(project="SIWY-25Z", job_type="tracin")

    artifact_root_dir = MODELS_DIR / dataset
    artifact_root_dir.mkdir(parents=True, exist_ok=True)

    # --- ŁADOWANIE CHECKPOINTU ---
    if not USE_LOCAL:
        # TODO: improve loading multiple epochs
        for epoch in [0, 1, 2, 3, 5, 7]:
            artifact = run.use_artifact(DOG_AND_CAT_MODEL_ARTIFACT_TEMPLATE.format(epoch), type="model")

            artifact_root_dir_epoch = artifact_root_dir / f"epoch_{epoch}"
            if not artifact_root_dir_epoch.exists():  # TODO: fix this check
                artifact_root_dir_epoch.mkdir(parents=True, exist_ok=True)
                artifact.download(root=artifact_root_dir_epoch)

        # TODO: remove this
        ckpt_files = list(Path(artifact_root_dir).glob("epoch_7/*.pt"))
        ckpt_files = ckpt_files[:1]
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

    # --- DANE ---

    dog_cat_ds = torch.load(PROCESSED_DATA_DIR / "dog-and-cat.pt", weights_only=False)
    airplane_ds = torch.load(AIRPLANES_DATASET_PATH, weights_only=False)
    logger.debug(f"Airplanes dataset: {airplane_ds}")
    train_ds = dog_cat_ds["train"]
    test_ds = airplane_ds["test"]
    # TODO: fix this
    test_ds = LabelToIdxWrapper(
        base_ds=test_ds, class_to_idx={"Cat": 0, "Dog": 1, "Airplane": 1}, transform=DEFAULT_TRANSFORM
    )

    logger.debug(f"Test dataset size: {len(test_ds)}")

    # Wymuś transformację na podzbiorach jeśli trzeba
    if hasattr(train_ds.dataset, "transform"):
        train_ds.dataset.transform = DEFAULT_TRANSFORM

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- TRACIN ---
    criterion = CrossEntropyLoss(label_smoothing=0.0, reduction="none")

    matrix = vectorized_calculate_tracin_score(
        model=model,
        criterion=criterion,
        weights_paths=weights_paths,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        lr=0.001,
        device=DEVICE,
        use_nested_loop_for_dot_product=False,
        float_labels=False,
    )

    # --- ZAPISZ MACIERZ DO WANDB ---
    torch.save(matrix, "tracin_score_matrix.pt")
    artifact = wandb.Artifact(f"tracin-matrix-{dataset}", type="result")
    artifact.add_file("tracin_score_matrix.pt")
    run.log_artifact(artifact)

    plot_tracin_top_contributors(run, train_loader, test_loader, matrix, test_idx=0, top_k=TOP_K)


def plot_no_train():
    run = wandb.init(project="SIWY-25Z", job_type="tracin-wyniki")

    dog_cat_ds = torch.load(PROCESSED_DATA_DIR / "dog-and-cat.pt", weights_only=False)
    airplane_ds = torch.load(AIRPLANES_DATASET_PATH, weights_only=False)
    logger.debug(f"Airplanes dataset: {airplane_ds}")
    train_ds = dog_cat_ds["train"]
    test_ds = airplane_ds["test"]
    # TODO: fix this
    test_ds = LabelToIdxWrapper(
        base_ds=test_ds, class_to_idx={"Cat": 0, "Dog": 1, "Airplane": 1}, transform=DEFAULT_TRANSFORM
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    plot_tracin_top_contributors(
        run, train_loader, test_loader, tracin_matrix=torch.load("tracin_score_matrix.pt"), test_idx=0, top_k=TOP_K
    )
    run.finish()


if __name__ == "__main__":
    main()
    # plot_no_train()
