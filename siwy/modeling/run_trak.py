from datetime import datetime
import os
import pathlib
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
import torch
from torch import Tensor
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from trak import TRAKer
from trak.savers import MmapSaver
import typer
import wandb

from siwy.common import DEVICE, denormalize
from siwy.config import FIGURES_DIR, IS_WINDOWS, MODELS_DIR, PROCESSED_DATA_DIR
from siwy.datasets.CatDogConfig import CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE, CLASS_TO_IDX
from siwy.datasets.common import DEFAULT_TRANSFORM, load_dataset
from siwy.datasets.wrapper import LabelToIdxWrapper
from siwy.ModelsFactory import construct_rn18


# Monkeypatch MmapSaver.init_store to fix Windows file locking issue
def patched_init_store(self, model_id) -> None:
    prefix = self.save_dir.joinpath(str(model_id))
    if os.path.exists(prefix):
        self.logger.info(f"Model ID folder {prefix} already exists")
    os.makedirs(prefix, exist_ok=True)
    featurized_so_far = np.zeros(shape=(self.train_set_size,), dtype=np.int32)
    ft = self._load(
        prefix.joinpath("_is_featurized.mmap"),
        shape=(self.train_set_size,),
        mode="w+",
        dtype=np.int32,
    )
    if ft is not None:
        ft[:] = featurized_so_far[:]
        ft.flush()
        # Explicitly release the file handle
        del ft

    self.load_current_store(model_id, mode="w+")


if IS_WINDOWS:
    pathlib.PosixPath = pathlib.WindowsPath
    MmapSaver.init_store = patched_init_store

app = typer.Typer()
DATETIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
TRAINING_PATH = PROCESSED_DATA_DIR / "trak" / DATETIME
CKPTS_PATH = TRAINING_PATH / "checkpoints"
RESULTS_PATH = MODELS_DIR / "trak_results" / DATETIME
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
AIRPLANES_DATASET_PATH = PROCESSED_DATA_DIR / "airplanes.pt"


def get_dataloader(ds, batch_size=256, num_workers=8, shuffle=False):
    assert ds is not None, "Dataset must be provided to create DataLoader."
    loader = torch.utils.data.DataLoader(dataset=ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    return loader


# TODO: move to common utils with plot_tracin_top_contributors
def plot_trak(run, ds_train: ImageFolder, ds_val: ImageFolder, scores: Tensor, top_k=5, dataset="dog-and-cat"):
    summary_table = wandb.Table(columns=["test_id", "train_id", "score"])
    for i in range(len(ds_val)):
        fig, axs = plt.subplots(ncols=7, figsize=(15, 3))
        fig.suptitle("Top scoring TRAK images from the train set")

        axs[0].imshow(denormalize(ds_val[i][0].permute(1, 2, 0)).clamp(0, 1))

        axs[0].axis("off")
        axs[0].set_title("Target image")
        axs[1].axis("off")
        logger.info(f"val class {ds_val[i][1]}")
        top_trak_scorers = scores[:, i].argsort()[-top_k:][::-1]

        # TODO: verify if this is correct and needed?
        # Add to summary table
        # scores is tensor, so we use cpu().tolist() or just tolist() if it's already on cpu/numpy
        if isinstance(scores, torch.Tensor):
            # Ensure indices and scores are list
            trak_scorers_list = top_trak_scorers.cpu().tolist()
            scores_list = scores[top_trak_scorers].cpu().tolist()
        else:
            # Assuming numpy
            trak_scorers_list = top_trak_scorers.tolist()
            scores_list = scores[top_trak_scorers].tolist()

        summary_table.add_data(i, trak_scorers_list, scores_list)

        logger.info(f"Test idx: {i}, top indices: {top_trak_scorers}, scores: {scores[top_trak_scorers]}")
        for ii, train_im_ind in enumerate(top_trak_scorers):
            logger.info(f"train id ({train_im_ind}): {ds_train[train_im_ind][1]}")
            axs[ii + 2].imshow(denormalize(ds_train[train_im_ind][0].permute(1, 2, 0)))
            axs[ii + 2].axis("off")
        logger.info("=" * 40)
        fig.show()
        plt.savefig(FIGURES_DIR / f"trak_{dataset}_val_image_{i}.png")
        run.log({"trak_results": wandb.Image(fig)})

    run.log({f"trak_{dataset}_scores": summary_table})


# @app.command()
def main(
    # model=typer.Option(
    #     "resnet18-pretrained",
    #     help="The training method to use. Options are: " + ", ".join(MODELS.keys()),
    # ),
    # dataset=typer.Option(, help="Name of the dataset to process"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    num_classes: int = typer.Option(2, help="Number of classes in the dataset"),
    epochs: list[int] = typer.Option(None, help="List of epochs to evaluate"),
    top_k: int = typer.Option(5, help="Number of top contributors to plot"),
):
    # TODO: use typer options for all params
    batch_size = 32
    num_classes = 3
    dataset = "dog-and-cat"
    ood_dataset = "airplanes"

    if epochs is None:
        epochs = [0, 1, 2, 4, 6, 8]
    model = (
        construct_rn18(num_classes=num_classes, weights=None).to(memory_format=torch.channels_last).to(DEVICE).eval()
    )

    # TODO: setup wandb config
    with wandb.init(project="SIWY-25Z", job_type="trak") as run:
        run.config.update(
            {
                "model": "resnet18-pretrained",
                "dataset": dataset,
                "batch_size": batch_size,
                "num_classes": num_classes,
                "ood_dataset": ood_dataset,
                "epochs": epochs,
                "top_k": 5,
            }
        )
        artifact_root_dir = MODELS_DIR / dataset
        artifact_root_dir.mkdir(parents=True, exist_ok=True)

        # --- LOAD CHECKPOINTS ---
        for epoch in epochs:
            artifact = run.use_artifact(CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE.format(epoch), type="model")

            artifact_root_dir_epoch = artifact_root_dir / f"epoch_{epoch}"
            if not artifact_root_dir_epoch.exists():  # TODO: fix this check, should skip downloading if already exists
                # TODO: check artifact/epoch instead of epoch_N ?
                artifact_root_dir_epoch.mkdir(parents=True, exist_ok=True)
                artifact.download(root=artifact_root_dir_epoch)

        ckpt_files = sorted(list(Path(artifact_root_dir).glob("**/*.pt")))
        logger.debug(f"ckpt_files: {ckpt_files}")
        assert len(ckpt_files) > 0, "No checkpoint found in artifact!"

        ckpts = [torch.load(ckpt, map_location=DEVICE) for ckpt in ckpt_files]
        logger.debug(f"Loaded {len(ckpts)} checkpoints for evaluation.")

        # --- DATA ---
        dog_cat_ds = load_dataset(dataset)
        airplane_ds = load_dataset(ood_dataset)
        logger.debug(f"Airplanes dataset: {airplane_ds}")

        logger.debug(f"Airplanes dataset: {airplane_ds}")
        train_ds = dog_cat_ds["train"]
        test_ds = airplane_ds["test"]
        test_ds = LabelToIdxWrapper(base_ds=test_ds, class_to_idx=CLASS_TO_IDX, transform=DEFAULT_TRANSFORM)

        if hasattr(train_ds.dataset, "transform"):
            train_ds.dataset.transform = DEFAULT_TRANSFORM

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        logger.info("Starting TRAK evaluation...")

        (RESULTS_PATH / "scores").mkdir(parents=True, exist_ok=True)

        logger.info(f"Train set size: {len(train_ds)}")

        # --- TRAK ---
        traker = TRAKer(
            model=model,
            task="image_classification",
            proj_dim=4096,
            train_set_size=len(train_ds),
            device=DEVICE,
            use_half_precision=True,
            save_dir=RESULTS_PATH,
        )

        logger.info(f"ckpts: {type(ckpts)} with length {len(ckpts)}")
        for model_id, ckpt in enumerate(tqdm(ckpts)):
            traker.load_checkpoint(ckpt, model_id=model_id)
            for batch in tqdm(train_loader):
                batch = [x.to(DEVICE) for x in batch]
                traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        traker.finalize_features()

        for model_id, ckpt in enumerate(tqdm(ckpts)):
            traker.start_scoring_checkpoint(
                exp_name="quickstart", checkpoint=ckpt, model_id=model_id, num_targets=len(test_loader.dataset)
            )
            for batch in test_loader:
                batch = [x.to(DEVICE) for x in batch]
                traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name="quickstart")
        _scores = open_memmap(RESULTS_PATH / "scores" / "quickstart.mmap")

        # --- SAVE SCORES TO WANDB ---
        scores_artifact = wandb.Artifact(
            name=f"trak-{dataset}",
            type="trak-scores",
        )
        scores_artifact.add_file(RESULTS_PATH / "scores" / "quickstart.mmap")

        # --- PLOT RESULTS ---
        if "run" in locals():
            run.log_artifact(scores_artifact)
            plot_trak(run, train_ds, test_ds, scores, top_k)
        else:
            logger.warning("Wandb run not initialized, skipping artifact logging and plotting.")

    logger.success("Trak finished successfully!")


if __name__ == "__main__":
    main()
