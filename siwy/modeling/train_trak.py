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

from siwy.config import PROCESSED_DATA_DIR, WANDB_DATASET_PATH, WANDB_PROJECT
from siwy.datasets.transform_and_upload_dataset import DATASETS
import wandb

app = typer.Typer()
DATETIME = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
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
    lr=0.4,
    epochs=24,
    momentum=0.9,
    weight_decay=5e-4,
    lr_peak_epoch=5,
    label_smoothing=0.0,
    model_id=0,
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

    for ep in range(epochs):
        print(ep)
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

        if ep in [12, 15, 18, 21, 23]:
            artifact_path = CKPTS_PATH / f"sd_{model_id}_epoch_{ep}.pt"
            torch.save(model.state_dict(), artifact_path)
            artifact = wandb.Artifact(
                name=f"trak-{DATETIME}-model-{model_id}-epoch-{ep}",
                type="model",
            )
            artifact.add_file(artifact_path)
            run.log_artifact(artifact)

    return model


def validate(model, val_loader):
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

    logger.info(f"Accuracy: {total_correct / total_num * 100:.1f}%")


def plot_trak(run, ds_train: ImageFolder, ds_val: ImageFolder, scores: Tensor):
    for i in [7, 21, 22]:
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
def main(
    model=typer.Option(
        "resnet18-pretrained",
        help="The training method to use. Options are: " + ", ".join(MODELS.keys()),
    ),
    dataset=typer.Option(Literal[*DATASETS], help="Name of the dataset to process"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    num_classes: int = typer.Option(2, help="Number of classes in the dataset"),
):
    # TODO: setup wandb config
    # start wandb run
    with wandb.init(project=f"{WANDB_PROJECT}", job_type="training") as run:
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
        logger.info("Loaded data for training.")

        # construct model
        model_fn = MODELS.get(model, None)
        assert model_fn is not None, f"Model {model} not found in MODELS."
        model = model_fn(num_classes=num_classes)
        logger.info(f"Constructed model: {model}")

        # train models
        for i in tqdm(range(1), desc="Training models.."):
            model = model.to(memory_format=torch.channels_last).cuda()
            model = train_model(run, model, loader_for_training, model_id=i)

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

        scores_artifact = wandb.Artifact(
            name=f"trak-{DATETIME}-scores",
            type="trak-scores",
        )
        scores_artifact.add_file(RESULTS_PATH / "scores" / "quickstart.mmap")
        run.log_artifact(scores_artifact)
        plot_trak(train_ds, val_test_ds, scores)


if __name__ == "__main__":
    app()
