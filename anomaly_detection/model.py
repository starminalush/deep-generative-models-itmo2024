import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn, optim
from torch.autograd import Variable

from submodules.MNAD.model.Reconstruction import convAE


class AnomalyDetection(L.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = convAE(3, memory_size=10, feature_dim=512, key_dim=512)
        self._m_items = F.normalize(
            torch.rand((10, 512), dtype=torch.float), dim=1
        ).cuda()  # Initialize the memory items
        self.loss_func_mse = nn.MSELoss(reduction="none")

    def training_step(self, batch, batch_idx):
        image = Variable(batch)
        outputs, separateness_loss, compactness_loss = self.forward(batch)
        loss_pixel = torch.mean(self.loss_func_mse(outputs, image))
        loss = loss_pixel + 0.1 * compactness_loss + 0.1 * separateness_loss
        self.log("train_loss", loss)
        return loss

    def forward(self, image):
        (
            outputs,
            _,
            _,
            self._m_items,
            softmax_score_query,
            softmax_score_memory,
            separateness_loss,
            compactness_loss,
        ) = self._model.forward(image, self._m_items, True)
        return outputs, separateness_loss, compactness_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = list(self._model.encoder.parameters()) + list(self._model.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=2e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        image = Variable(batch)
        outputs, separateness_loss, compactness_loss = self.forward(batch)
        loss_pixel = torch.mean(self.loss_func_mse(outputs, image))
        loss = loss_pixel + 0.1 * compactness_loss + 0.1 * separateness_loss
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        image = Variable(batch)
        outputs, separateness_loss, compactness_loss = self.forward(batch)
        loss_pixel = torch.mean(self.loss_func_mse(outputs, image))
        loss = loss_pixel + 0.1 * compactness_loss + 0.1 * separateness_loss
        self.log("test_loss", loss)
