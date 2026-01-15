from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from siwy.config import FIGURES_DIR, WANDB_ORG, WANDB_PROJECT
from siwy.modeling.common_configs import COMPARISON_RUN_IDS
import wandb


def get_runs_by_ids(run_ids: list[str]):
    """Fetch specified runs from wandb by their IDs."""
    api = wandb.Api()

    data = []
    for run_id in run_ids:
        try:
            run = api.run(f"{WANDB_ORG}/{WANDB_PROJECT}/{run_id}")

            config = run.config
            summary = run.summary._json_dict
            runtime_seconds = summary.get("_runtime", None)
            runtime_minutes = runtime_seconds / 60 if runtime_seconds else None
            method = run.job_type if run.job_type else "unknown"

            data.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "method": method,
                    "dataset": config.get("dataset", "unknown"),
                    "model": config.get("model", "unknown"),
                    "ood_dataset": config.get("ood_dataset", "unknown"),
                    "top_k": config.get("top_k", None),
                    "epochs": str(config.get("epochs", "unknown")),
                    "batch_size": config.get("batch_size", None),
                    "runtime_minutes": runtime_minutes,
                    "tags": run.tags,
                }
            )

            logger.info(f"Loaded run: {run.name} ({method}) - ID: {run_id}")

        except Exception as e:
            logger.error(f"Failed to load run {run_id}: {e}")

    return pd.DataFrame(data)


def create_comparison_table_image(df, output_path):
    display_df = df[["method", "dataset", "model", "batch_size", "epochs", "runtime_minutes"]].copy()

    display_df["runtime_minutes"] = display_df["runtime_minutes"].round(2)

    display_df.columns = ["Method", "Dataset", "Model", "Batch Size", "Epochs", "Runtime (min)"]

    display_df = display_df.sort_values(["Method", "Dataset"])
    fig, ax = plt.subplots(figsize=(14, len(display_df) * 0.5 + 2))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.20, 0.18, 0.12, 0.15, 0.12],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title("Runtime Comparison Across Methods and Datasets", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved table image to {output_path}")

    return output_path


def create_bar_charts_per_dataset(df, output_dir):
    """Create separate bar charts for each dataset."""
    datasets = df["dataset"].unique()

    for dataset in datasets:
        df_dataset = df[df["dataset"] == dataset].copy()

        df_dataset = df_dataset.sort_values("method")

        plt.figure(figsize=(10, 6))

        ax = sns.barplot(
            data=df_dataset, x="method", y="runtime_minutes", hue="method", palette="Set2", dodge=False, legend=False
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f min", padding=3)

        dataset_title = dataset.replace("-", " ").title()

        plt.title(f"Runtime Comparison - {dataset_title}", fontsize=14, fontweight="bold")
        plt.xlabel("Method", fontsize=12)
        plt.ylabel("Runtime (minutes)", fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()

        safe_dataset_name = dataset.replace("-", "_")
        output_path = output_dir / f"runtime_comparison_{safe_dataset_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved chart for {dataset} to {output_path}")


def create_combined_bar_chart(df, output_path):
    """Create a combined bar chart with all datasets for overview."""
    plt.figure(figsize=(14, 6))

    plt.title("Runtime Comparison - All Datasets", fontsize=14, fontweight="bold")
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Runtime (minutes)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved combined chart to {output_path}")


def main(
    run_ids: list[str] = typer.Option(None, help="Run IDs to compare (uses config file if not specified)"),
):
    """Compare runtimes of specified runs."""
    if run_ids is None or len(run_ids) == 0:
        run_ids = COMPARISON_RUN_IDS
        logger.info(f"Using run IDs from config file: {len(run_ids)} runs")

    if not run_ids:
        logger.error("No run IDs provided. Please add run IDs to siwy/runtime_comparison_config.py")
        return

    # Create output directory
    output_dir = FIGURES_DIR / "runtime-compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    logger.info(f"Fetching {len(run_ids)} runs from wandb...")
    df = get_runs_by_ids(run_ids)

    if df.empty:
        logger.error("No runs found. Check your run IDs.")
        return

    logger.info(f"Successfully loaded {len(df)} runs")

    csv_path = output_dir / "wandb_runs_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved raw data to {csv_path}")

    table_path = output_dir / "runtime_comparison_table.png"
    combined_chart_path = output_dir / "runtime_comparison_combined.png"

    create_comparison_table_image(df, table_path)
    create_bar_charts_per_dataset(df, output_dir)
    create_combined_bar_chart(df, combined_chart_path)

    logger.success(f"Analysis complete! All files saved to {output_dir}")
    logger.info(f"Created {len(df['dataset'].unique())} individual dataset charts")

    with wandb.init(project=WANDB_PROJECT, job_type="analysis", name="runtime-comparison") as run:
        run.config.update({"comparison_run_ids": run_ids})

        run.log(
            {
                "runtime_table": wandb.Image(table_path),
                "runtime_combined": wandb.Image(combined_chart_path),
            }
        )

        for dataset in df["dataset"].unique():
            safe_dataset_name = dataset.replace("-", "_")
            chart_path = output_dir / f"runtime_comparison_{safe_dataset_name}.png"
            run.log({f"runtime_{safe_dataset_name}": wandb.Image(chart_path)})

        artifact = wandb.Artifact("runtime-comparison", type="analysis")
        artifact.add_file(table_path)
        artifact.add_file(combined_chart_path)
        artifact.add_file(csv_path)
        for dataset in df["dataset"].unique():
            safe_dataset_name = dataset.replace("-", "_")
            artifact.add_file(output_dir / f"runtime_comparison_{safe_dataset_name}.png")
        run.log_artifact(artifact)

    logger.success("Analysis completed and uploaded to wandb")


if __name__ == "__main__":
    typer.run(main)
