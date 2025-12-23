from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
from models.ModelsFactory import construct_rn18
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from TracInPyTorch.src.tracin import vectorized_calculate_tracin_score

from siwy.config import PROCESSED_DATA_DIR, PROJ_ROOT
import wandb

WANDB_MODEL_ARTIFACT = "jarcin/SIWY-25Z/cat-dog-2025-12-23-17-17-44-model-0-epoch-7:v0"
LOCAL_CKPT_PATH = PROJ_ROOT / "artifacts" / "cat-dog-2025-12-23-17-17-44-model-0-epoch-7-v0"

AIRPLANES_DATASET_PATH = PROCESSED_DATA_DIR / "airplanes.pt"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
USE_LOCAL = False


def plot_tracin_top_contributors(run, train_loader, test_loader, tracin_matrix, test_idx=0, top_k=5):
    # Pobierz obrazy z loaderów
    train_imgs = []
    for batch in train_loader:
        ims, _ = batch
        train_imgs.extend([im for im in ims])
    test_imgs = []
    for batch in test_loader:
        ims, _ = batch
        test_imgs.extend([im for im in ims])
    scores = tracin_matrix[:, test_idx]
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


def main():
    model = construct_rn18(num_classes=2).to(DEVICE)

    # --- ŁADOWANIE CHECKPOINTU ---
    if USE_LOCAL:
        # model.load_state_dict(torch.load(LOCAL_CKPT_PATH, map_location=DEVICE))
        print(f"Loaded local checkpoint: {LOCAL_CKPT_PATH}")
        # weights_paths = [str(LOCAL_CKPT_PATH)]
    else:
        # run = wandb.init()
        # artifact = run.use_artifact(WANDB_MODEL_ARTIFACT, type="model")
        # artifact_dir = artifact.download()
        artifact_dir = LOCAL_CKPT_PATH
        logger.info(f"Downloaded artifact to: {artifact_dir}")
        # Pobierz wszystkie checkpointy *.pt z artifactu (np. epoki 1-7)
        # ckpt_files = sorted([str(p) for p in Path(artifact_dir).rglob("*.pt") if any(f"epoch_{ep}" in str(p) for ep in range(1, 8))])
        ckpt_files = list(Path(artifact_dir).glob("*.pt"))
        logger.info(f"ckpt_files: {ckpt_files}")
        assert len(ckpt_files) > 0, "No checkpoint found in artifact!"
        # Załaduj pierwszy checkpoint do modelu (np. epoch_1)
        model.load_state_dict(torch.load(ckpt_files[0], map_location=DEVICE))
        print(f"Loaded checkpoint from wandb: {ckpt_files[0]}")
        weights_paths = ckpt_files
    return

    model.eval()

    # --- DANE ---
    ds = torch.load(AIRPLANES_DATASET_PATH, weights_only=False)
    train_ds = ds["train"]
    test_ds = ds["test"]

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Wymuś transformację na podzbiorach jeśli trzeba
    if hasattr(train_ds.dataset, "transform"):
        train_ds.dataset.transform = transform
    if hasattr(test_ds.dataset, "transform"):
        test_ds.dataset.transform = transform

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    # --- TRACIN ---
    criterion = CrossEntropyLoss(label_smoothing=0.0, reduction="none")
    # ckpt_files = sorted(list((DATA_DIR / Path("./checkpoints-2cls")).rglob("*.pt")))
    # weights_paths = [LOCAL_CKPT_PATH]  # lub lista checkpointów, jeśli chcesz
    weights_paths = sorted(str(p) for p in Path(LOCAL_CKPT_PATH).glob("*.pt"))

    matrix = vectorized_calculate_tracin_score(
        model=model,
        criterion=criterion,  # OK
        weights_paths=weights_paths,
        train_dataloader=train_loader,  # OK
        test_dataloader=test_loader,  # OK
        lr=0.001,  # OK
        device=DEVICE,  # OK
        use_nested_loop_for_dot_product=False,  # OK
        float_labels=False,  # OK
    )

    # --- ZAPISZ MACIERZ DO WANDB ---
    run = wandb.init(project="SIWY-25Z", job_type="tracin")
    torch.save(matrix, "tracin_score_matrix.pt")
    artifact = wandb.Artifact("tracin-matrix", type="result")
    artifact.add_file("tracin_score_matrix.pt")
    run.log_artifact(artifact)

    plot_tracin_top_contributors(run, train_loader, test_loader, matrix, test_idx=0, top_k=TOP_K)


if __name__ == "__main__":
    main()
