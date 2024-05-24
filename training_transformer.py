

import os
from matplotlib import pyplot as plt
from torch import optim, nn, utils, Tensor
import torch
from torchvision.transforms import ToTensor
import lightning as L
from torchvision import models
import torch.nn.functional as F
from mde.network.depth_decoder import MDEModel
from mde.network.discriminator import Discriminator
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_msssim import ssim
from vit_pytorch.na_vit import NaViT
class MDE(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.mde_unet = MDEModel()
            
        self.discriminator = Discriminator()
        self.automatic_optimization = False
        self.loss=nn.BCELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        torch.set_float32_matmul_precision('highest')

    def generator_loss(self,disc_generated_output, gen_output, target):
        gan_loss = self.loss(disc_generated_output,torch.ones_like(disc_generated_output))
        # Mean absolute error
        l1_loss = self.compute_reprojection_loss(gen_output,target )
        total_gen_loss = gan_loss + (0.5 * l1_loss)

        return total_gen_loss
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)

        l1_loss = abs_diff.mean(1, True)
        
       
        # ssim_loss = torch.concat([(1-ssim(img[0].unsqueeze(0), img[1].unsqueeze(0))).unsqueeze(0)*torch.ones_like(img[0][0]).unsqueeze(0).unsqueeze(0) for img in zip(pred,target)],dim=1)
        ssim_loss=1-ssim(pred, target)
        loss = 0.35 * ssim_loss + 0.65 * l1_loss

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
        with torch.autograd.set_detect_anomaly(True):
            g_opt, d_opt = self.optimizers()

            x, y = batch
            z = self.mde_unet(x.unsqueeze(0))
            # Optimize Discriminator 
            
            d_x = self.discriminator(torch.concat([x, y], dim=1))
            d_z = self.discriminator(torch.concat([x, z], dim=1))
            errD=self.discriminator_loss(d_x,d_z)
           
            d_opt.zero_grad()
            self.manual_backward(errD,retain_graph=True)
            d_opt.step()

            # Optimize Generator 
            errG = self.generator_loss(d_z.detach().clone(), z, y)
            g_opt.zero_grad()
            self.manual_backward(errG)
            g_opt.step()

            self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        
        z = self.mde_unet(x.unsqueeze(0))
  
        x_d=self.discriminator(torch.concat([ x,y], dim=1))
        z_d=self.discriminator(torch.concat([ x,z], dim=1))
        errD=self.discriminator_loss(x_d,z_d)

        g_loss = self.generator_loss(z_d.detach().clone(), z, y)

        self.log_dict({"g_val_loss": g_loss, "d_val_loss": errD}, prog_bar=True)
        return g_loss
    
        
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.mde_unet.parameters(), lr=1e-3)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        return g_opt, d_opt
