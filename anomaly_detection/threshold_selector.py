from pathlib import (
    Path,
)

import click
import torch
import torch.nn.functional as F
from dataset import (
    DefectsDataset,
)
from lightning import (
    seed_everything,
)
from matplotlib import (
    pyplot as plt,
)
from model import (
    AnomalyDetection,
)
from torch.utils.data import (
    DataLoader,
)
from transforms import (
    get_test_transforms,
)

torch.manual_seed(42)
seed_everything(42, workers=True)


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
    print(mse_losses)
    plt.hist(mse_losses, bins=30, density=True, alpha=0.5, color="b")
    plt.title("Loss Distribution")
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.savefig("reports/mse.png")


if __name__ == "__main__":
    select_threshold()
