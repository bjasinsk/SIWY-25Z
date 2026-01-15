from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import torch
import typer

from siwy.common import MEAN, STD, denormalize
from siwy.config import FIGURES_DIR, REPORTS_DIR
from siwy.datasets.common import load_dataset
from siwy.modeling.modeling_utils import setup_windows_compatibility
import wandb

setup_windows_compatibility()

app = typer.Typer()


def get_one_method_scores(run, method_name, dataset, artifact_type=None, is_torch=True, filename=None):
    WANDB_PATH = f"jarcin/SIWY-25Z/{method_name}-{dataset}:latest"
    SCORES_DIR = REPORTS_DIR / "scores"

    if artifact_type is None:
        artifact_type = f"{method_name}-scores"

    artifact = run.use_artifact(WANDB_PATH, type=artifact_type)

    if is_torch:
        if filename is None:
            filename = f"{method_name}_score_matrix.pt"
        local_artifact_path = artifact.get_entry(filename).download(root=SCORES_DIR)
        result = torch.load(local_artifact_path).detach().cpu().numpy()
    else:
        if filename is None:
            filename = "quickstart.mmap"
        local_artifact_path = artifact.get_entry(filename).download(root=SCORES_DIR)
        result = np.load(local_artifact_path, mmap_mode="r")

    return result


def count_unique_topk(scores, top_k=5, num_test_samples=5):
    all_topk_indices = []

    for test_idx in range(min(num_test_samples, scores.shape[1])):
        test_scores = scores[:, test_idx]
        top_k_indices = np.argsort(test_scores)[-top_k:]
        all_topk_indices.extend(top_k_indices.tolist())

    unique_indices = set(all_topk_indices)
    total_possible = top_k * num_test_samples

    return {
        "Unique": len(unique_indices),
        "Total": total_possible,
        "Percentage": (len(unique_indices) / total_possible * 100) if total_possible > 0 else 0,
        "Unique_Indices": list(unique_indices),
    }


def save_table_as_image(table, output_path, title):
    fig, ax = plt.subplots(figsize=(10, len(table._rows) * 0.5 + 2))
    ax.axis("tight")
    ax.axis("off")

    table_data = [table.field_names] + table._rows

    mpl_table = ax.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.25, 0.20, 0.20, 0.25])

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    mpl_table.scale(1, 2)

    for i in range(len(table.field_names)):
        mpl_table[(0, i)].set_facecolor("#68A142")
        mpl_table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(table_data)):
        for j in range(len(table.field_names)):
            if i % 2 == 0:
                mpl_table[(i, j)].set_facecolor("#D0D7CC")

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved table image to {output_path}")


def plot_unique_images(dataset_name, method_name, unique_indices, train_dataset, output_path):
    num_images = len(unique_indices)

    if num_images == 0:
        logger.warning(f"No unique images to plot for {method_name} on {dataset_name}")
        return

    cols = 5
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, train_idx in enumerate(sorted(unique_indices)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        img, label = train_dataset[train_idx]

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)
            img = denormalize(img, mean=MEAN, std=STD)
            img = img.numpy()
            img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Train #{train_idx}", fontsize=8)

    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.suptitle(
        f"Unique Training Images - {method_name.upper()} - {dataset_name.replace('-', ' ').title()}\n"
        f"({num_images} unique out of max 25)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved unique images plot to {output_path}")


@app.command()
def analyze_uniqueness(
    datasets: list[str] = typer.Option(
        ["dog-and-cat", "bus-and-truck-easy-train", "horse-and-elephant-easy-train"], help="List of datasets to analyze"
    ),
    methods: list[str] = typer.Option(["dualda", "trak", "tracin"], help="List of methods to analyze"),
    top_k: int = typer.Option(5, help="Number of top contributors per test image"),
    num_test_samples: int = typer.Option(5, help="Number of test samples to analyze"),
    output_format: str = typer.Option("both", help="Output format: table, csv, or both"),
    plot_images: bool = typer.Option(True, help="Plot unique training images"),
):
    results = []

    output_dir = FIGURES_DIR / "uniqueness-compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    with wandb.init(project="SIWY-25Z", job_type="analyze-uniqueness") as run:
        for dataset in datasets:
            logger.info(f"\nProcessing dataset: {dataset}")

            ds = load_dataset(dataset)
            train_dataset = ds["train"]

            table = PrettyTable()
            table.field_names = ["Method", "Unique Images", "Total Slots", "Percentage"]

            method_unique_indices = {}

            for method in methods:
                try:
                    if method == "dualda":
                        scores = get_one_method_scores(run, method, dataset, artifact_type="dualda-scores")
                    elif method == "trak":
                        scores = get_one_method_scores(run, method, dataset, is_torch=False)
                    elif method == "tracin":
                        scores = get_one_method_scores(
                            run,
                            method,
                            dataset,
                            artifact_type="tracin-scores",
                            filename=f"tracin_score_matrix_{dataset}.pt",
                        )
                    else:
                        logger.warning(f"Unknown method: {method}")
                        continue

                    stats = count_unique_topk(scores, top_k=top_k, num_test_samples=num_test_samples)
                    method_unique_indices[method] = stats["Unique_Indices"]

                    table.add_row([method.upper(), stats["Unique"], stats["Total"], f"{stats['Percentage']:.1f}%"])

                    results.append(
                        {
                            "Dataset": dataset,
                            "Method": method.upper(),
                            "Top-K": top_k,
                            "Test Samples": num_test_samples,
                            "Unique": stats["Unique"],
                            "Total": stats["Total"],
                            "Percentage": stats["Percentage"],
                        }
                    )

                except Exception as e:
                    logger.warning(f"Could not load {method} scores for {dataset}: {e}")
                    table.add_row([method.upper(), "N/A", "N/A", "N/A"])
                    method_unique_indices[method] = []

            if output_format in ["table", "both"]:
                logger.info(f"\nUniqueness Statistics for {dataset} (top-{top_k}, {num_test_samples} test samples):")
                print(table)

                output_path = REPORTS_DIR / f"uniqueness_{dataset}_top{top_k}_test{num_test_samples}.txt"
                with open(output_path, "w") as f:
                    f.write(f"Uniqueness Statistics for {dataset}\n")
                    f.write(f"Top-{top_k} contributors, {num_test_samples} test samples\n\n")
                    f.write(str(table))
                logger.info(f"Saved table to {output_path}")

                image_path = output_dir / f"uniqueness_{dataset}_top{top_k}_test{num_test_samples}.png"
                save_table_as_image(
                    table,
                    image_path,
                    f"Unique Training Images in Top-{top_k} - {dataset.replace('-', ' ').title()}\n({num_test_samples} test samples)",
                )

            if plot_images:
                for method, unique_indices in method_unique_indices.items():
                    if unique_indices:
                        image_plot_path = (
                            output_dir / f"unique_images_{dataset}_{method}_top{top_k}_test{num_test_samples}.png"
                        )
                        plot_unique_images(dataset, method, unique_indices, train_dataset, image_plot_path)

    if output_format in ["csv", "both"]:
        df = pd.DataFrame(results)

        csv_path = REPORTS_DIR / f"uniqueness_top{top_k}_test{num_test_samples}.csv"
        df.to_csv(csv_path, index=False)
        logger.success(f"Saved combined statistics to {csv_path}")

        pivot_unique = df.pivot(index="Dataset", columns="Method", values="Unique")
        pivot_path_unique = REPORTS_DIR / f"uniqueness_count_top{top_k}_test{num_test_samples}.csv"
        pivot_unique.to_csv(pivot_path_unique)
        logger.info(f"Saved unique count comparison to {pivot_path_unique}")

        pivot_pct = df.pivot(index="Dataset", columns="Method", values="Percentage")
        pivot_path_pct = REPORTS_DIR / f"uniqueness_percentage_top{top_k}_test{num_test_samples}.csv"
        pivot_pct.to_csv(pivot_path_pct)
        logger.info(f"Saved percentage comparison to {pivot_path_pct}")

    logger.success(f"\nAll uniqueness tables and images saved in: {output_dir}")


if __name__ == "__main__":
    app()
