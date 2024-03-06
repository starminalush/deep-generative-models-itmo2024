import lightning as L
import numpy as np
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


class GenerateCallback(L.Callback):
    """Generate grid comparing original and reconstructed images."""

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs
        self._every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if trainer.current_epoch % self._every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)
            pl_module.eval()
            with torch.no_grad():
                reconst_imgs = pl_module(input_imgs)[0]
            pl_module.train()
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True)
            wandb_logger = trainer.logger.experiment
            wandb_logger.log({"reconstructed_images": [wandb.Image(grid)]})


class LatentSpaceVisCallback(L.Callback):
    """Build visualization of latent space."""

    def __init__(self, dataloader, every_n_epochs):
        super().__init__()
        self._dataloader = dataloader
        self._every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if trainer.current_epoch % self._every_n_epochs == 0:
            pl_module.eval()
            features_list = []
            with torch.no_grad():
                for img in self._dataloader:
                    features = pl_module._model.encoder(img.to(pl_module.device)).detach().cpu().numpy()
                    features_list.append(features)
            pl_module.train()
            features_list = np.concatenate(features_list, axis=0)
            num_examples, channels, height, width = features_list.shape
            features_list = features_list.reshape(num_examples, -1)
            plt.figure(figsize=(7, 7))
            plt.title("Latent Space Visualization PCA")
            pca = PCA(n_components=2, random_state=42)
            cluster_data = pca.fit_transform(features_list)
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c="blue", marker="o", alpha=0.5)
            wandb_logger = trainer.logger.experiment
            wandb_logger.log({"latent space": [wandb.Image(plt)]})
