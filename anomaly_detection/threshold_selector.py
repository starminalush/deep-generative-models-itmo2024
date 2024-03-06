from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from dataset import DefectsDataset
from lightning import seed_everything
from matplotlib import pyplot as plt
from model import AnomalyDetection
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from transforms import get_test_transforms

torch.manual_seed(42)
seed_everything(42, workers=True)


def _visualize_thresholds(
    mse_losses: list[float], predicted_labels: list[int], threshold: float, cluster_centers: list[list[float]]
):
    # Визуализация
    plt.scatter(
        range(len(mse_losses)),
        mse_losses,
        c=predicted_labels,
        cmap="viridis",
        label="Clusters",
    )
    plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
    plt.scatter(
        range(len(cluster_centers)),
        cluster_centers,
        marker="X",
        c="red",
        s=200,
        label="Cluster Centers",
    )
    plt.xlabel("Sample Index")
    plt.ylabel("MSE Loss")
    plt.title("Clustering and Threshold Visualization")
    plt.legend()
    plt.savefig("threshold_vis.png")


def _build_clusters(mse_losses):
    optimal_k = 2

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(mse_losses)

    cluster_centers = kmeans.cluster_centers_
    return cluster_centers


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path))
@click.option("--checkpoint-path", type=click.Path(path_type=Path))
@click.option("--img-size", type=int)
def select_threshold(dataset_path: Path | str, checkpoint_path: Path | str, img_size: int) -> None:
    """Select threshold by mse on proliv.
    Args:
        dataset_path: Dataset 'proliv' location.
        checkpoint_path: Trained model checkpoint path.
        img_size: Image size. Needed for transforms.
    """
    model = AnomalyDetection.load_from_checkpoint(checkpoint_path)
    model.eval()
    mse_losses = list()
    proliv_dataset = DefectsDataset(dataset_path, transform=get_test_transforms(img_size))
    proliv_dataloader = DataLoader(proliv_dataset, batch_size=1)
    with torch.no_grad():
        for image in proliv_dataloader:
            result = model(image.to("cuda"))
            reconstacted_image = result[0]
            mse_losses.append(F.mse_loss(image, reconstacted_image.cpu()).item())
    mse_losses_reshaped = np.asarray(mse_losses).reshape(-1, 1)

    cluster_centers = _build_clusters(mse_losses_reshaped)
    threshold = np.mean(cluster_centers)
    predicted_labels = mse_losses >= threshold

    _visualize_thresholds(mse_losses, predicted_labels, threshold, cluster_centers)

    with open("params.yaml", "w") as file:
        yaml.dump({"threshold": float(threshold)}, file, default_flow_style=False)


if __name__ == "__main__":
    select_threshold()
