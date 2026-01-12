import torch
import wandb
import matplotlib.pyplot as plt

from siwy.config import FIGURES_DIR

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def denormalize(img, mean=None, std=None):
    if std is None:
        std = STD
    if mean is None:
        mean = MEAN
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return img * std.view(1, 1, 3) + mean.view(1, 1, 3)


GENERATOR = torch.manual_seed(42)
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def plot_explainability_results(
    run, ds_train, ds_val, scores, logger, method_name, top_k=5, dataset_name="dog-and-cat"
):
    summary_table = wandb.Table(columns=["test_id", "train_id", "score"])
    for i in range(len(ds_val)):
        fig, axs = plt.subplots(ncols=7, figsize=(15, 3))
        fig.suptitle(f"Top scoring {method_name} images from the train set")

        axs[0].imshow(denormalize(ds_val[i][0].permute(1, 2, 0)).clamp(0, 1))
        axs[0].axis("off")
        axs[0].set_title("Target image")
        axs[1].axis("off")
        logger.info(f"val class {ds_val[i][1]}")
        top_trak_scorers = scores[:, i].argsort()[-top_k:][::-1]
        trak_scorers_list = top_trak_scorers.tolist()
        scores_list = scores[top_trak_scorers].tolist()

        summary_table.add_data(i, trak_scorers_list, scores_list)

        for ii, train_im_ind in enumerate(top_trak_scorers):
            logger.info(f"train id ({train_im_ind}): {ds_train[train_im_ind][1]}")
            axs[ii + 2].imshow(denormalize(ds_train[train_im_ind][0].permute(1, 2, 0)))
            axs[ii + 2].axis("off")
        logger.info("=" * 40)
        fig.show()
        plt.savefig(FIGURES_DIR / f"trak_{dataset_name}_val_image_{i}.png")
        run.log({"trak_results": wandb.Image(fig)})

    run.log({f"trak_{dataset_name}_scores": summary_table})


def plot_multi_method_explainability(
    run, ds_train, ds_val, scores_list, method_names, top_k=5, dataset_name="dog-and-cat"
):
    num_methods = len(method_names)  # Should be 3 based on your request

    for i in range(len(ds_val)):
        # 1. Define the Layout
        # 'T' is the Target Image.
        # Row index 'r' and Column index 'c' define the training images.
        layout = []
        for r in range(num_methods):
            row = ["Target"] + [f"m{r}_c{c}" for c in range(top_k)]
            layout.append(row)

        fig, axd = plt.subplot_mosaic(layout, figsize=(3 * (top_k + 1), 3 * num_methods), constrained_layout=True)

        # 2. Plot Target Image (Spans all rows on the left)
        target_img = denormalize(ds_val[i][0].permute(1, 2, 0)).clamp(0, 1)
        axd["Target"].imshow(target_img)
        axd["Target"].set_title(f"Target Image", fontsize=16, fontweight="bold")
        axd["Target"].axis("off")

        # 3. Plot Each Method's Row
        for m_idx, scores in enumerate(scores_list):
            method_name = method_names[m_idx]

            # Get top indices for this specific method
            top_indices = scores[:, i].argsort()[-top_k:][::-1]

            for col_idx, train_im_ind in enumerate(top_indices):
                ax_key = f"m{m_idx}_c{col_idx}"
                train_img = denormalize(ds_train[train_im_ind][0].permute(1, 2, 0)).clamp(0, 1)

                axd[ax_key].imshow(train_img)
                axd[ax_key].axis("off")

                # Only add the Method Title to the first image in the row
                if col_idx == 0:
                    axd[ax_key].set_title(f"{method_name}", loc="left", fontsize=14, fontweight="bold", color="blue")

                # Optional: Add score as subtitle
                score_val = scores[train_im_ind, i]
                axd[ax_key].text(
                    0.5, -0.1, f"Score: {score_val:.3f}", transform=axd[ax_key].transAxes, ha="center", fontsize=10
                )

        plt.savefig(FIGURES_DIR / f"method_comparison_{dataset_name}_val_image_{i}.png")
        run.log({"comparison": wandb.Image(fig)})
