from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import torch
import typer

from siwy.config import FIGURES_DIR, REPORTS_DIR
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


def compute_topk_statistics(scores, top_k=5):
    num_train, num_test = scores.shape

    all_topk_scores = []
    for test_idx in range(num_test):
        test_scores = scores[:, test_idx]
        top_k_indices = np.argsort(test_scores)[-top_k:]
        top_k_scores = test_scores[top_k_indices]
        all_topk_scores.extend(top_k_scores)

    all_topk_scores = np.array(all_topk_scores)

    return {
        "Min": np.min(all_topk_scores),
        "Max": np.max(all_topk_scores),
        "Mean": np.mean(all_topk_scores),
        "Median": np.median(all_topk_scores),
        "Std Dev": np.std(all_topk_scores),
        "Num Test Images": num_test,
    }


def save_table_as_image(table, output_path, title):
    """Save PrettyTable as PNG image."""
    fig, ax = plt.subplots(figsize=(12, len(table._rows) * 0.5 + 2))
    ax.axis("tight")
    ax.axis("off")

    table_data = [table.field_names] + table._rows

    mpl_table = ax.table(
        cellText=table_data, cellLoc="center", loc="center", colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15]
    )

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


@app.command()
def analyze_topk_scores(
    datasets: list[str] = typer.Option(
        ["dog-and-cat", "bus-and-truck-easy-train", "horse-and-elephant-easy-train"], help="List of datasets to analyze"
    ),
    methods: list[str] = typer.Option(["dualda", "trak", "tracin"], help="List of methods to analyze"),
    top_k: int = typer.Option(5, help="Number of top contributors per test image"),
    output_format: str = typer.Option("both", help="Output format: table, csv, or both"),
):
    results = []

    output_dir = FIGURES_DIR / "score-compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    with wandb.init(project="SIWY-25Z", job_type="analyze-topk-scores") as run:
        for dataset in datasets:
            logger.info(f"\nProcessing dataset: {dataset}")

            table = PrettyTable()
            table.field_names = ["Method", "Min", "Max", "Mean", "Median", "Std Dev", "Test Images"]

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

                    stats = compute_topk_statistics(scores, top_k=top_k)

                    table.add_row(
                        [
                            method.upper(),
                            f"{stats['Min']:.6f}",
                            f"{stats['Max']:.6f}",
                            f"{stats['Mean']:.6f}",
                            f"{stats['Median']:.6f}",
                            f"{stats['Std Dev']:.6f}",
                            stats["Num Test Images"],
                        ]
                    )

                    results.append(
                        {
                            "Dataset": dataset,
                            "Method": method.upper(),
                            "Top-K": top_k,
                            **{k: v for k, v in stats.items()},
                        }
                    )

                except Exception as e:
                    logger.warning(f"Could not load {method} scores for {dataset}: {e}")
                    table.add_row([method.upper(), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

            if output_format in ["table", "both"]:
                logger.info(f"\nTop-{top_k} Statistics for {dataset}:")
                print(table)

                output_path = REPORTS_DIR / f"topk_statistics_{dataset}_top{top_k}.txt"
                with open(output_path, "w") as f:
                    f.write(f"Top-{top_k} Statistics for {dataset}\n")
                    f.write(str(table))
                logger.info(f"Saved table to {output_path}")

                image_path = output_dir / f"topk_statistics_{dataset}_top{top_k}.png"
                save_table_as_image(
                    table, image_path, f"Top-{top_k} Score Statistics - {dataset.replace('-', ' ').title()}"
                )

    if output_format in ["csv", "both"]:
        df = pd.DataFrame(results)

        csv_path = REPORTS_DIR / f"topk_statistics_top{top_k}.csv"
        df.to_csv(csv_path, index=False)
        logger.success(f"Saved combined statistics to {csv_path}")

        for metric in ["Min", "Max", "Mean", "Median", "Std Dev"]:
            pivot = df.pivot(index="Dataset", columns="Method", values=metric)
            pivot_path = REPORTS_DIR / f"topk_{metric.lower().replace(' ', '_')}_top{top_k}.csv"
            pivot.to_csv(pivot_path)
            logger.info(f"Saved {metric} comparison to {pivot_path}")

    logger.success(f"\nAll table images saved in: {output_dir}")


if __name__ == "__main__":
    app()
