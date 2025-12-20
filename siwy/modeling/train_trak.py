from datetime import datetime

from loguru import logger
from matplotlib import pyplot as plt
from models.ModelsFactory import MODELS
import numpy as np
from numpy.lib.format import open_memmap
import torch
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from trak import TRAKer
import typer
from typing_extensions import Literal
import wandb

from siwy.config import PROCESSED_DATA_DIR, WANDB_DATASET_PATH, WANDB_PROJECT
from siwy.datasets.transform_and_upload_dataset import DATASETS

app = typer.Typer()
DATETIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
TRAINING_PATH = PROCESSED_DATA_DIR / "trak" / DATETIME
CKPTS_PATH = TRAINING_PATH / "checkpoints"
RESULTS_PATH = TRAINING_PATH / "results"

GENERATOR = torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader(ds, batch_size=256, num_workers=8, shuffle=False):
    assert ds is not None, "Dataset must be provided to create DataLoader."
    loader = torch.utils.data.DataLoader(dataset=ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    return loader


def train_model(
    run,
    model,
    loader,
    val_loader=None,
    lr=0.4,
    epochs=24,
    momentum=0.9,
    weight_decay=5e-4,
    lr_peak_epoch=5,
    label_smoothing=0.0,
    model_id=0,
    validate_every_n=3,
    early_stop_patience=5,
    checkpoint_interval=5,
    last_k_epochs=30,
):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle
    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    # Calculate checkpoint epochs: working backwards from final epoch
    checkpoint_epochs = list(range(epochs, max(0, epochs - last_k_epochs) - 1, -checkpoint_interval))

    # Early stopping tracking
    best_val_accuracy = 0.0
    patience_counter = 0

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for it, (ims, labs) in enumerate(loader):
            ims = ims.cuda()
            labs = labs.cuda()
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Log average training loss per epoch
        avg_loss = epoch_loss / num_batches
        run.log({"train_loss": avg_loss, "epoch": ep})
        logger.info(f"Epoch {ep}: avg_loss={avg_loss:.4f}")

        # Periodic validation
        if val_loader is not None and (ep + 1) % validate_every_n == 0:
            val_accuracy = validate(model, val_loader, run=run, epoch=ep)

            # Check for improvement
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                logger.info(f"New best validation accuracy: {best_val_accuracy:.2f}%")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")

                # Early stopping
                if patience_counter >= early_stop_patience:
                    logger.warning(f"Early stopping triggered at epoch {ep}")
                    run.log({"early_stop_epoch": ep, "reason": "validation accuracy plateau"})
                    break

        # Save checkpoints at specified epochs
        if ep in checkpoint_epochs:
            artifact_path = CKPTS_PATH / f"{DATETIME}-sd_{model_id}_epoch_{ep}.pt"
            torch.save(model.state_dict(), artifact_path)
            artifact = wandb.Artifact(
                name=f"trak-{DATETIME}-model-{model_id}-epoch-{ep}",
                type="model",
            )
            artifact.add_file(artifact_path)
            run.log_artifact(artifact)
            logger.info(f"Saved checkpoint at epoch {ep}")

    return model


def validate(model, val_loader, run=None, epoch=None):
    model.eval()

    with torch.no_grad():
        total_correct, total_num = 0.0, 0.0
        for ims, labs in tqdm(val_loader):
            ims = ims.cuda()
            labs = labs.cuda()
            with autocast():
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

    accuracy = total_correct / total_num * 100
    logger.info(f"Accuracy: {accuracy:.1f}%")

    if run and epoch is not None:
        run.log({"val_accuracy": accuracy, "epoch": epoch})

    return accuracy


def plot_trak(run, ds_train: ImageFolder, ds_val: ImageFolder, scores: Tensor, image_ids: list[int]):
    for i in image_ids:
        fig, axs = plt.subplots(ncols=7, figsize=(15, 3))
        fig.suptitle("Top scoring TRAK images from the train set")

        axs[0].imshow(ds_val[i][0].permute(1, 2, 0))

        axs[0].axis("off")
        axs[0].set_title("Target image")
        axs[1].axis("off")
        logger.info(f"val class {ds_val[i][1]}")
        top_trak_scorers = scores[:, i].argsort()[-5:][::-1]
        for ii, train_im_ind in enumerate(top_trak_scorers):
            logger.info(f"train id ({train_im_ind}): {ds_train[train_im_ind][1]}")
            axs[ii + 2].imshow(ds_train[train_im_ind][0].permute(1, 2, 0))
            axs[ii + 2].axis("off")
        logger.info("=" * 40)
        fig.show()
        plt.savefig(RESULTS_PATH / f"trak_val_image_{i}.png")
        run.log({"trak_results": wandb.Image(fig)})


@app.command()
def train(
    model=typer.Option(
        "resnet18-pretrained",
        help="The training method to use. Options are: " + ", ".join(MODELS.keys()),
    ),
    dataset=typer.Option(Literal[*DATASETS], help="Name of the dataset to process"),
    batch_size: int = typer.Option(64, help="Batch size for training"),
    num_classes: int = typer.Option(2, help="Number of classes in the dataset"),
    # Training hyperparameters
    lr: float = typer.Option(0.4, help="Learning rate"),
    epochs: int = typer.Option(24, help="Number of training epochs"),
    momentum: float = typer.Option(0.9, help="SGD momentum"),
    weight_decay: float = typer.Option(5e-4, help="Weight decay"),
    lr_peak_epoch: int = typer.Option(5, help="Epoch at which learning rate peaks"),
    label_smoothing: float = typer.Option(0.0, help="Label smoothing factor"),
    # Validation & early stopping
    validate_every_n_epochs: int = typer.Option(3, help="Run validation every N epochs"),
    early_stop_patience: int = typer.Option(5, help="Stop if no improvement for N validation checks"),
    # Checkpoint configuration
    checkpoint_interval: int = typer.Option(5, help="Save every N epochs within the last K epochs"),
    last_k_epochs: int = typer.Option(30, help="Only save checkpoints in the last K epochs"),
    # TRAK visualization
    trak_plot_image_ids: str = typer.Option("7,21,22", help="Comma-separated image IDs to visualize in TRAK plots"),
):
    with wandb.init(project=f"{WANDB_PROJECT}", job_type="training") as run:
        run.config.update(
            {
                "model": model,
                "dataset": dataset,
                "batch_size": batch_size,
                "num_classes": num_classes,
                "lr": lr,
                "epochs": epochs,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "lr_peak_epoch": lr_peak_epoch,
                "label_smoothing": label_smoothing,
                "validate_every_n_epochs": validate_every_n_epochs,
                "early_stop_patience": early_stop_patience,
                "checkpoint_interval": checkpoint_interval,
                "last_k_epochs": last_k_epochs,
                "trak_plot_image_ids": trak_plot_image_ids,
            }
        )

        artifact = run.use_artifact(WANDB_DATASET_PATH(dataset), type="dataset")
        artifact_path = artifact.download(PROCESSED_DATA_DIR)

        # prepare dataset
        ds = torch.load(f"{artifact_path}/{dataset}.pt", weights_only=False)
        train_ds = ds["train"]
        val_test_ds = ConcatDataset([ds["val"], ds["test"]])

        CKPTS_PATH.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        # get data loaders
        loader_for_training = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = get_dataloader(val_test_ds, batch_size=batch_size, shuffle=False)
        logger.info("Loaded data for training.")

        # construct model
        model_fn = MODELS.get(model, None)
        assert model_fn is not None, f"Model {model} not found in MODELS."
        model = model_fn(num_classes=num_classes)
        logger.info(f"Constructed model: {model}")

        # train models
        for i in tqdm(range(1), desc="Training models.."):
            model = model.to(memory_format=torch.channels_last).cuda()
            model = train_model(
                run=run,
                model=model,
                loader=loader_for_training,
                val_loader=val_loader,
                lr=lr,
                epochs=epochs,
                momentum=momentum,
                weight_decay=weight_decay,
                lr_peak_epoch=lr_peak_epoch,
                label_smoothing=label_smoothing,
                model_id=i,
                validate_every_n=validate_every_n_epochs,
                early_stop_patience=early_stop_patience,
                checkpoint_interval=checkpoint_interval,
                last_k_epochs=last_k_epochs,
            )

        logger.success("Training complete.")

        ckpt_files = sorted(list(CKPTS_PATH.rglob("*.pt")))
        ckpts = [torch.load(ckpt, map_location="cpu") for ckpt in ckpt_files]

        # validate, get model accuracy
        validate(model, get_dataloader(val_test_ds, batch_size=batch_size))

        ## TRAK evaluation
        logger.info("Starting TRAK evaluation...")
        loader_train = get_dataloader(train_ds, batch_size=batch_size)

        traker = TRAKer(
            model=model, task="image_classification", proj_dim=4096, train_set_size=len(loader_train.dataset)
        )

        for model_id, ckpt in enumerate(tqdm(ckpts)):
            traker.load_checkpoint(ckpt, model_id=model_id)
            for batch in tqdm(loader_train):
                batch = [x.cuda() for x in batch]
                traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        traker.finalize_features()

        loader_targets = get_dataloader(val_test_ds, batch_size=batch_size)

        for model_id, ckpt in enumerate(tqdm(ckpts)):
            traker.start_scoring_checkpoint(
                exp_name="quickstart", checkpoint=ckpt, model_id=model_id, num_targets=len(loader_targets.dataset)
            )
            for batch in loader_targets:
                batch = [x.cuda() for x in batch]
                traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name="quickstart")
        _scores = open_memmap(RESULTS_PATH / "scores" / "quickstart.mmap")

        # Verify scores shape
        expected_shape = (len(train_ds), len(val_test_ds))
        assert scores.shape == expected_shape, f"Scores shape {scores.shape} != expected {expected_shape}"

        run.log({"scores_shape_train": scores.shape[0], "scores_shape_test": scores.shape[1]})
        logger.info(
            f"Scores shape verified: {scores.shape} (train_size={scores.shape[0]}, test_size={scores.shape[1]})"
        )

        scores_artifact = wandb.Artifact(
            name=f"trak-{DATETIME}-scores",
            type="trak-scores",
        )
        scores_artifact.add_file(RESULTS_PATH / "scores" / "quickstart.mmap")
        run.log_artifact(scores_artifact)

        # Parse image IDs for TRAK visualization
        image_ids = [int(x.strip()) for x in trak_plot_image_ids.split(",")]
        plot_trak(run, train_ds, val_test_ds, scores, image_ids)


@app.command()
def plot_scores(
    dataset: str = typer.Option(..., help="Name of the dataset to use"),
    scores_path: str = typer.Option(..., help="Path to the scores .mmap file"),
    trak_plot_image_ids: str = typer.Option("7,21,22", help="Comma-separated image IDs to visualize"),
    use_wandb: bool = typer.Option(False, help="Whether to log to wandb"),
):
    """Generate TRAK plots from pre-computed scores without training."""
    output_dir = PROCESSED_DATA_DIR / "trak" / "plots" / DATETIME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scores
    logger.info(f"Loading scores from {scores_path}")
    scores = open_memmap(scores_path, mode="r")
    logger.info(f"Loaded scores with shape: {scores.shape}")

    # Load dataset
    if use_wandb:
        run = wandb.init(project=f"{WANDB_PROJECT}", job_type="plotting")
        artifact = run.use_artifact(WANDB_DATASET_PATH(dataset), type="dataset")
        artifact_path = artifact.download(PROCESSED_DATA_DIR)
    else:
        run = None
        artifact_path = PROCESSED_DATA_DIR / dataset

    ds = torch.load(f"{artifact_path}/{dataset}.pt", weights_only=False)
    train_ds = ds["train"]
    val_test_ds = ConcatDataset([ds["val"], ds["test"]])

    logger.info(f"Loaded dataset: train_size={len(train_ds)}, val_test_size={len(val_test_ds)}")

    # Verify scores shape
    expected_shape = (len(train_ds), len(val_test_ds))
    assert scores.shape == expected_shape, f"Scores shape {scores.shape} != expected {expected_shape}"

    # Parse image IDs
    image_ids = [int(x.strip()) for x in trak_plot_image_ids.split(",")]
    logger.info(f"Generating plots for images: {image_ids}")

    # Override RESULTS_PATH for this command
    global RESULTS_PATH
    original_results_path = RESULTS_PATH
    RESULTS_PATH = output_dir

    # Generate plots
    plot_trak(run, train_ds, val_test_ds, scores, image_ids)

    # Restore RESULTS_PATH
    RESULTS_PATH = original_results_path

    logger.success(f"Plots saved to {output_dir}")

    if use_wandb:
        run.finish()


if __name__ == "__main__":
    app()

"""
uv run train --model resnet18-pretrained --dataset dog-and-cat --batch_size 32 --num_classes 2 --lr 0.4 --epochs 24 --momentum 0.9 --weight_decay 5e-4 --lr_peak_epoch 5 --label_smoothing 0.0 --validate_every_n_epochs 3 --early_stop_patience 5 --checkpoint_interval 5 --last_k_epochs 30 --trak_plot_image_ids "7,21,22"

uv run plot_scores --dataset dog-and-cat --scores_path /path/to/scores.mmap --batch_size 32 --num_classes 2 --trak_plot_image_ids "7,21,22" --use_wandb
"""
