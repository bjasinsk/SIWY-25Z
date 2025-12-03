from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import wandb

from siwy.config import MODELS_DIR, PROCESSED_DATA_DIR, WANDB_PROJECT

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    with wandb.init(project=WANDB_PROJECT, config={}) as run:
        # ---- REPLACE THIS WITH YOUR OWN CODE ----
        logger.info("Performing inference for model...")
        for i in tqdm(range(10), total=10):
            if i == 5:
                logger.info("Something happened for iteration 5.")
            run.log({"iteration": i})
        logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
