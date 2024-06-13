import os
from matplotlib import pyplot as plt
from torch import optim, nn, utils, Tensor
import torch
from torchvision.transforms import ToTensor
import lightning as L
from torchvision import models
import torch.nn.functional as F
from mde.network.depth_decoder import MDEModel,UNet
from mde.network.discriminator import Discriminator
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_msssim import ssim
from torchvision.transforms.functional import resize

class MDE(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.mde_unet = UNet(1)
        self.discriminator = Discriminator()
        self.automatic_optimization = False
        self.loss=nn.BCELoss()
        self.mseLoss = nn.MSELoss()

        
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def generator_loss(self, gen_output, target):
        l1_loss = self.compute_reprojection_loss(gen_output,target )
        total_gen_loss = l1_loss
        return total_gen_loss
        
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        # l1_loss = torch.mean(torch.abs(target- pred.where(target!=0,torch.tensor(0.0))),1)
        l2_loss = self.mseLoss(target, pred.where(target!=0,torch.tensor(0.0)))
        ssim_loss=1-ssim(pred.where(target!=0,torch.tensor(0.0)), target)
        loss = 0.6 * ssim_loss + 0.65 * l2_loss

        return torch.mean(loss)

    def discriminator_loss(self,disc_real_output, disc_generated_output):
        real_loss = self.loss( disc_real_output,torch.ones_like(disc_real_output))
        generated_loss = self.loss(disc_generated_output,torch.zeros_like(disc_generated_output))
        total_disc_loss = real_loss + generated_loss

        # return torch.mean(total_disc_loss)
        return (total_disc_loss)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        g_opt = self.optimizers()

        x, y = batch
        z = self.mde_unet(x)
        z=resize(z,[100,332])
        y=resize(y,[100,332])
        
        errG = self.generator_loss(z, y)
        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log_dict({"g_loss": errG}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch        
        z = self.mde_unet(x)
        z=resize(z,[100,332])
        y=resize(y,[100,332])

        g_loss = self.generator_loss( z, y)

        self.log_dict({"g_val_loss": g_loss}, prog_bar=True)
        return g_loss

        
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.mde_unet.parameters(), lr=1e-5)
        return g_opt

class MyCallback(L.Callback):
    def __init__(self,reference_image):
        super().__init__()
        self.reference_image=reference_image
        
    def on_train_epoch_end(self, trainer, pl_module):
        tensorboard = trainer.logger.experiment
        
        z = pl_module.mde_unet(self.reference_image.to('cuda').unsqueeze(0))
        tensorboard.add_image("val",z.squeeze(0).squeeze(0),trainer.current_epoch,dataformats="HW")

