import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torchvision import models

from mde.network.depth_decoder import MDEModel

class MDE(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.mde_unet = MDEModel()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.mde_unet(x)
        loss = nn.functional.mse_loss(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer