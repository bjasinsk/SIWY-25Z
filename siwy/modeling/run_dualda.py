import pathlib
from pathlib import Path

from dualxda.explainers import DualDA
from loguru import logger
import torch
import torch.nn as nn
import wandb
import typer

from siwy.common import DEVICE, plot_explainability_results
from siwy.config import IS_WINDOWS, MODELS_DIR, PROJ_ROOT
from siwy.datasets.CatDogConfig import CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE, CLASS_TO_IDX
from siwy.datasets.BusTruckConfig import BUS_AND_TRUCK_MODEL_ARTIFACT_TEMPLATE
from siwy.datasets.HorseElephantConfig import HORSE_AND_ELEPHANT_MODEL_ARTIFACT_TEMPLATE
from siwy.datasets.common import DEFAULT_TRANSFORM, load_dataset
from siwy.modeling.common_configs import  DATASET_CONFIGS
from siwy.datasets.wrapper import LabelToIdxWrapper
from siwy.ModelsFactory import construct_rn18

if IS_WINDOWS:
    pathlib.PosixPath = pathlib.WindowsPath

class DualDAModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

        self.classifier = original_model.fc

        self.features = nn.Sequential(*list(original_model.children())[:-1], nn.Flatten())

    def forward(self, x):
        return self.original_model(x)

app = typer.Typer()


@app.command()
def main(    
    dataset: str = typer.Option(
        "dog-and-cat", help="Dataset: dog-and-cat | bus-and-truck-easy-train | horse-and-elephant-easy-train"
    ),
    ood_dataset: str = typer.Option("airplanes", help="Out-of-distribution dataset for testing"),
    batch_size: int = typer.Option(16, help="Batch size for data loading"),
    epochs: list[int] = typer.Option(None, help="Epochs to evaluate (uses dataset defaults if not specified)"),
    top_k: int = typer.Option(5, help="Number of top contributors to visualize"),
    lr: float = typer.Option(0.001, help="Learning rate used during training (for TracIn calculation)")):

    # Get dataset configuration
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset]
    if epochs is None:
        epochs = [config["default_epochs"][-1]]
    num_classes = config["num_classes"]
    artifact_template = config["artifact_template"]

    model = construct_rn18(num_classes=num_classes, weights=None).to(DEVICE)
    with wandb.init(project="SIWY-25Z", job_type="dualda") as run:
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
            }
        )

        artifact_root_dir = MODELS_DIR / dataset
        artifact_root_dir.mkdir(parents=True, exist_ok=True)

        # --- LOAD CHECKPOINTS ---

        for epoch in epochs:
            artifact = run.use_artifact(artifact_template.format(epoch), type="model")
            artifact_root_dir_epoch = artifact_root_dir / f"epoch_{epoch}"

            if not artifact_root_dir_epoch.exists():
                artifact_root_dir_epoch.mkdir(parents=True, exist_ok=True)
                artifact.download(root=artifact_root_dir_epoch)

        ckpt_files = sorted(list(Path(artifact_root_dir).glob("**/*.pt")))
        logger.debug(f"ckpt_files: {ckpt_files}")
        assert len(ckpt_files) > 0, "No checkpoint found in artifact!"
        # Załaduj pierwszy checkpoint do modelu (np. epoch_1)
        model.load_state_dict(torch.load(ckpt_files[0], map_location=DEVICE))
        print(f"Loaded checkpoint from wandb: {ckpt_files[0]}")

        # --- DATA ---
        ds = load_dataset(dataset)
        ood_ds = load_dataset(ood_dataset)
        logger.debug(f"Out of distribution dataset: {ood_ds}")
        train_ds = ds["train"]
        test_ds = ood_ds["test"]
        test_ds = LabelToIdxWrapper(base_ds=test_ds, class_to_idx=CLASS_TO_IDX, transform=DEFAULT_TRANSFORM)

        logger.debug(f"Test dataset size: {len(test_ds)}")

        if hasattr(train_ds.dataset, "transform"):
            train_ds.dataset.transform = DEFAULT_TRANSFORM

        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

        # --- DUALDA ---

        C = 0.01
        cache_dir = "./content/cache_dir"
        features_dir = "./content/features_dir"
        model_wrapped = DualDAModelWrapper(model)

        explainer = DualDA(
            model_wrapped,
            train_ds,
            device=DEVICE,
            dir=cache_dir,
            features_dir=features_dir,
            normalize=True,
            C=C,
        )

        explainer.train()
        result = list()

        for example in test_loader:
            preds = model(example[0]).argmax(dim=-1)
            contribution = explainer.explain(example[0], preds)
            result.append(contribution)

        result = torch.stack(result, dim=1)

        # --- SAVE SCORES TO WANDB ---
        matrix_path = MODELS_DIR / "dualda_score_matrix.pt"
        torch.save(result, matrix_path)
        artifact = wandb.Artifact(f"dualda-{dataset}", type="dualda-scores")
        artifact.add_file(matrix_path)
        run.log_artifact(artifact)

        # --- PLOT RESULTS ---
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        plot_explainability_results(run, train_ds, test_ds, result, logger, "DualDA", top_k)

    logger.success("DualDA finished successfully!")


if __name__ == "__main__":
    app()

"""
Usage from root directory:
# dog-and-cat
uv run siwy/modeling/run_dualda.py --dataset "dog-and-cat" --ood-dataset "airplanes"

# bus and truck
uv run siwy/modeling/run_dualda.py --dataset "bus-and-truck-easy-train" --ood-dataset "airplanes"

# horse and elephant 
uv run siwy/modeling/run_dualda.py --dataset "horse-and-elephant-easy-train" --ood-dataset "airplanes"
"""
