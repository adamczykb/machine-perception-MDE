import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.transforms import ToTensor
import lightning as L
from torchvision import models
import torch.nn.functional as F
from ignite.metrics import SSIM
from mde.network.depth_decoder import MDEModel
from mde.network.discriminator import Discriminator

class MDE(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.mde_unet = MDEModel()
        self.discriminator = Discriminator()
        self.automatic_optimization = False
        self.loss=nn.BCELoss()
        self.ssim = SSIM(data_range=1.0)

    def generator_loss(self,disc_generated_output, gen_output, target):
        gan_loss = self.loss(torch.ones_like(disc_generated_output), disc_generated_output)
        # Mean absolute error
        l1_loss = torch.mean(torch.abs(target - gen_output))
        total_gen_loss = gan_loss + (0.5 * l1_loss)

        return total_gen_loss
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def discriminator_loss(self,disc_real_output, disc_generated_output):
        real_loss = self.loss(torch.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss(torch.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        with torch.autograd.set_detect_anomaly(False):
            g_opt, d_opt = self.optimizers()

            x, y = batch
            z = self.mde_unet(x)

            
            # Optimize Discriminator 
            
            d_x = self.discriminator(torch.concat([ x,y], dim=1))
            d_z = self.discriminator(torch.concat([ x,z], dim=1))
            
            errD=self.discriminator_loss(d_x,d_z)
            print(errD)
            d_opt.zero_grad()
            self.manual_backward(errD,retain_graph=True)
            d_opt.step()
            

            # Optimize Generator 
            
            errG = self.generator_loss(d_z.detach(), z, y)
            print(errD)

            g_opt.zero_grad()
            self.manual_backward(errG)
            g_opt.step()

            self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.mde_unet(x)
        x_d=self.discriminator(torch.concat([ x,y], dim=1))
        z_d=self.discriminator(torch.concat([ x,z], dim=1))

        l1_loss = torch.mean(torch.abs(z_d - x_d))

        self.log("val_loss", l1_loss,prog_bar=True)
        return l1_loss

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.mde_unet.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-5)
        return g_opt, d_opt
