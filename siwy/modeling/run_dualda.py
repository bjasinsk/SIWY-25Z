import pathlib
from pathlib import Path

from dualxda.explainers import DualDA
from loguru import logger
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

from siwy.common import DEVICE, plot_explainability_results
from siwy.config import IS_WINDOWS, MODELS_DIR, PROJ_ROOT
from siwy.datasets.CatDogConfig import CAT_AND_DOG_MODEL_ARTIFACT_TEMPLATE, CLASS_TO_IDX
from siwy.datasets.common import DEFAULT_TRANSFORM, load_dataset
from siwy.datasets.wrapper import LabelToIdxWrapper
from siwy.ModelsFactory import construct_rn18

if IS_WINDOWS:
    pathlib.PosixPath = pathlib.WindowsPath

# TODO: fix this file
LOCAL_CKPT_PATH = PROJ_ROOT / "artifacts" / "cat-dog-2025-12-23-17-17-44-model-0-epoch-7-v0"

USE_LOCAL = False

class DualDAModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
        self.classifier = original_model.fc
        
        self.features = nn.Sequential(*list(original_model.children())[:-1], nn.Flatten())

    def forward(self, x):
        return self.original_model(x)

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
            weights_paths = [str(LOCAL_CKPT_PATH)]  # noqa: F841
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
        artifact = wandb.Artifact(f"dualda-{dataset}", type="dudalda-scores")
        artifact.add_file(matrix_path)
        run.log_artifact(artifact)

        # --- PLOT RESULTS ---
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        plot_explainability_results(run, train_ds, test_ds, result, logger, 'DualDA', top_k)

    logger.success("DualDA finished successfully!")



if __name__ == "__main__":
    main()
