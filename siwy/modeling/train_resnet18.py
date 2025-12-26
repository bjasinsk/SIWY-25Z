from datetime import datetime
import pathlib
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm
import typer
import wandb

from siwy.common import DEVICE
from siwy.config import IS_WINDOWS, PROCESSED_DATA_DIR, WANDB_DATASET_PATH, WANDB_PROJECT
from siwy.datasets.transform_and_upload_dataset import DEFAULT_TRANSFORM
from siwy.ModelsFactory import MODELS

if IS_WINDOWS:
    pathlib.PosixPath = pathlib.WindowsPath

app = typer.Typer()
DATETIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
TRAINING_PATH = PROCESSED_DATA_DIR / "tracin" / DATETIME
CKPTS_PATH = TRAINING_PATH / "checkpoints"
RESULTS_PATH = TRAINING_PATH / "results"


def get_dataloader(ds, batch_size=256, num_workers=8, shuffle=False):
    assert ds is not None, "Dataset must be provided to create DataLoader."
    if isinstance(ds, Subset):
        base_ds = ds.dataset
    else:
        base_ds = ds
    if hasattr(base_ds, "transform"):
        base_ds.transform = DEFAULT_TRANSFORM
    loader = torch.utils.data.DataLoader(dataset=ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return loader


def train_model(
    run,
    model,
    loader,
    loader_val,
    lr=0.001,
    epochs=200,
    momentum=0.9,
    weight_decay=5e-4,
    lr_peak_epoch=5,
    label_smoothing=0.0,
    model_id=0,
    max_worse_epochs=8,
):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)

    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    loss_fn = CrossEntropyLoss(label_smoothing=0.0)

    best_acc = 0.0
    worse_epochs = 0

    for ep in range(epochs):
        print(ep)
        epoch_loss = 0.0
        num_batches = 0
        model.train()
        for it, (ims, labs) in enumerate(loader):
            if it == 0 and ep == 0:
                print("Sample labels:", labs[:10])
                print("Labels dtype:", labs.dtype)
                print("Labels min/max:", labs.min().item(), labs.max().item())
            ims = ims.to(DEVICE)
            labs = labs.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(ims)
            loss = loss_fn(out, labs)
            epoch_loss += loss.item()
            num_batches += 1

            loss.backward()
            opt.step()
            scheduler.step()

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        run.log({"train_loss": avg_loss, "epoch": ep})

        if loader_val is not None:
            model.eval()
            correct, total = 0, 0
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for ims, labs in loader_val:
                    ims = ims.to(DEVICE)
                    labs = labs.to(DEVICE)
                    out = model(ims)
                    loss = loss_fn(out, labs)
                    val_loss += loss.item()
                    val_batches += 1
                    correct += out.argmax(1).eq(labs).sum().item()
                    total += ims.size(0)
            acc = correct / total if total > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            logger.info(f"Epoch {ep}: val accuracy = {acc * 100:.2f}%, val loss = {avg_val_loss:.4f}")
            run.log({"val accuracy": acc, "val_loss": avg_val_loss, "epoch": ep})

            if acc > best_acc:
                best_acc = acc
                worse_epochs = 0
                # save
                artifact_path = CKPTS_PATH / "best_model.pt"
                torch.save(model.state_dict(), artifact_path)
                artifact = wandb.Artifact(
                    name=f"cat-dog-{DATETIME}-model-{model_id}-epoch-{ep}",
                    type="model",
                )
                artifact.add_file(artifact_path)
                run.log_artifact(artifact)
            else:
                worse_epochs += 1
                logger.info("Worse epoch")
            if worse_epochs >= max_worse_epochs:
                logger.info(f"Early stopping at epoch {ep + 1}. Best val accuracy: {best_acc * 100:.2f}%")
                break

        # if ep % 10 == 0:
        #     artifact_path = CKPTS_PATH / f"sd_{model_id}_epoch_{ep}.pt"
        #     torch.save(model.state_dict(), artifact_path)
        #     artifact = wandb.Artifact(
        #         name=f"trak-{DATETIME}-model-{model_id}-epoch-{ep}",
        #         type="model",
        #     )
        #     artifact.add_file(artifact_path)
        #     run.log_artifact(artifact)

    return model


def validate(model, val_loader):
    model.eval()

    with torch.no_grad():
        total_correct, total_num = 0.0, 0.0
        for ims, labs in tqdm(val_loader):
            ims = ims.to(DEVICE)
            labs = labs.to(DEVICE)
            with autocast():
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

    logger.info(f"Accuracy: {total_correct / total_num * 100:.1f}%")


@app.command()
def main(
    model=typer.Option(
        "resnet18-pretrained",
        help="The training method to use. Options are: " + ", ".join(MODELS.keys()),
    ),
    dataset: str = typer.Option("dog-and-cat", help="Name of the dataset to process"),
    # dataset=typer.Option(Literal[*DATASETS], help="Name of the dataset to process"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    num_classes: int = typer.Option(3, help="Number of classes in the dataset"),
):
    # TODO: setup wandb config
    # start wandb run
    with wandb.init(project=f"{WANDB_PROJECT}", job_type="training") as run:
        artifact = run.use_artifact(WANDB_DATASET_PATH(dataset), type="dataset")
        artifact_path = artifact.download(PROCESSED_DATA_DIR)

        # prepare dataset
        # ds = torch.load(f"{artifact_path}/{dataset}.pt", weights_only=False)
        ds = torch.load(Path(artifact_path) / f"{dataset}.pt", weights_only=False)
        train_ds = ds["train"]
        val_test_ds = ConcatDataset([ds["val"], ds["test"]])

        CKPTS_PATH.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        # get data loaders
        loader_for_training = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
        loader_for_validation = get_dataloader(val_test_ds, batch_size=batch_size, shuffle=False)
        logger.info("Loaded data for training.")

        # construct model
        model_fn = MODELS.get(model, None)
        assert model_fn is not None, f"Model {model} not found in MODELS."
        model = model_fn(num_classes=num_classes)
        logger.info(f"Constructed model: {model}")

        # train models
        for i in tqdm(range(1), desc="Training models.."):
            model = model.to(memory_format=torch.channels_last).to(DEVICE)
            model = train_model(run, model, loader_for_training, loader_for_validation, model_id=i)

        logger.success("Training complete.")

        # validate, get model accuracy
        validate(model, get_dataloader(val_test_ds, batch_size=batch_size))


if __name__ == "__main__":
    app()
