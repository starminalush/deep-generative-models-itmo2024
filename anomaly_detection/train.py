from pathlib import (
    Path,
)

import click
import torch
import torch.nn.functional as F
from callbacks import (
    GenerateCallback,
)
from dataset import (
    DefectsDataset,
)
from lightning import (
    Trainer,
)
from lightning.pytorch import (
    seed_everything,
)
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import (
    WandbLogger,
)
from model import (
    AnomalyDetection,
)
from submodules.MNAD.model.Reconstruction import (
    convAE,
)
from torch.utils import (
    data,
)
from transforms import (
    get_train_transforms,
)

torch.manual_seed(42)
seed_everything(42, workers=True)


def _get_debug_images(num, dataset):
    return torch.stack([dataset[i] for i in range(num)], dim=0)


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path))
@click.option("--device", type=str, required=False, default="cuda")
@click.option("--batch-size", type=int)
@click.option("--img-size", type=int)
@click.option("--num-epochs", type=int)
@click.option("--project-name", type=str)
def train(
    dataset_path: Path | str,
    device: str = "cuda",
    batch_size: int = 256,
    img_size: int = 32,
    num_epochs: int = 100,
    project_name: str = "defects",
) -> None:
    """Train model for anomaly detection."""

    checkpoint_path = Path("models")
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    # args.c = 3
    # args.msize = 10
    # args.fdim = 512
    # args.mdim = 512

    model = AnomalyDetection()
    wandb_logger = WandbLogger(project=project_name)
    wandb_logger.watch(model)

    transform = get_train_transforms(img_size=img_size)

    train_dataset = DefectsDataset(dataset_path / "train", transform=transform, labels=None)
    val_dataset = DefectsDataset(dataset_path / "val", transform=transform, labels=None)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    trainer = Trainer(
        default_root_dir=checkpoint_path,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        logger=wandb_logger,
        deterministic=True,
        log_every_n_steps=2,
        max_epochs=num_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath="models",
                filename="model",
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
            ),
            GenerateCallback(_get_debug_images(8, dataset=val_dataset), every_n_epochs=4),
            LearningRateMonitor("epoch"),
        ],
    )

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader, verbose=False)


if __name__ == "__main__":
    train()
