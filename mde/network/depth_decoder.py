# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from mde.network.monodepth2 import *
from torchvision.transforms.functional import resize
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,stride=1, padding=1,  bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,stride=1, padding=1,  bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 =nn.Conv2d(in_channels, in_channels // 2, kernel_size=3,stride=1,padding=1,  bias=False)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = self.up1(x1)
        # x1 = self.up2(x1)
        
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x2 = resize(x2, x1.size()[2:4])
        
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class BottleNeckBlock(nn.Module):
    def __init__(
        self, in_channels, kernel_size=3, padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA1 = nn.Conv2d(in_channels,in_channels*2, kernel_size, strides,padding=1)
        self.convA2 = nn.Conv2d(in_channels*2,in_channels*2, kernel_size, strides,padding=1)
        # self.convB = nn.Conv2d(in_channels*2,in_channels, kernel_size, strides)
        self.reluA = nn.LeakyReLU()
        self.reluB = nn.LeakyReLU()

    def forward(self, xin):
        x = self.convA1(xin)
        x = self.reluA(x)
        x = self.convA2(x)
        x = self.reluB(x)

        # diffY = x.size()[2] - xin.size()[2]
        # diffX = x.size()[3] - xin.size()[3]
      
        
        # # xin = F.pad(xin, [diffX // 2, diffX - diffX // 2,
        # #                 diffY // 2, diffY - diffY // 2])
        # dec4 = torch.cat((x, xin), dim=1)
        # x = self.convB(dec4)
        # # x = self.convB(x)
        # x = self.reluB(x)
        return x
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.layers=[1024, 512,256,128,64]
        self.batch_norm=nn.ModuleList([nn.BatchNorm2d(l) for l in self.layers])
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(3, self.layers[4], kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(self.layers[4], self.layers[4], kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(self.layers[4], self.layers[3], kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(self.layers[3], self.layers[3], kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(self.layers[3], self.layers[2], kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(self.layers[2], self.layers[2], kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(self.layers[2], self.layers[1], kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(self.layers[1], self.layers[1], kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(self.layers[1], self.layers[0], kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(self.layers[0], self.layers[0], kernel_size=3, padding=1) # output: 28x28x1024
        


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(self.layers[0], self.layers[1], kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(self.layers[0], self.layers[1], kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(self.layers[1], self.layers[1], kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(self.layers[1], self.layers[2], kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(self.layers[1], self.layers[2], kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(self.layers[2], self.layers[2], kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(self.layers[2], self.layers[3], kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(self.layers[2], self.layers[3], kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(self.layers[3], self.layers[3], kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(self.layers[3], self.layers[4], kernel_size=2, stride=2)
        self.d41 = nn.Conv2d( self.layers[3],  self.layers[4], kernel_size=3, padding=1)
        self.d42 = nn.Conv2d( self.layers[4],  self.layers[4], kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d( self.layers[4], n_class, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xe12=self.batch_norm[4](xe12)
        xp1 = self.pool1(xe12)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        xe22=self.batch_norm[3](xe22)
        xp2 = self.pool2(xe22)

        xe31 =self.relu(self.e31(xp2))
        xe32 = self.relu(self.e32(xe31))
        xe32=self.batch_norm[2](xe32)
        xp3 = self.pool3(xe32)

        xe41 = self.relu(self.e41(xp3))
        xe42 = self.relu(self.e42(xe41))
        xe42=self.batch_norm[1](xe42)
        xp4 = self.pool4(xe42)

        xe51 =self.relu(self.e51(xp4))
        xe52 = self.relu(self.e52(xe51))
        xe53=self.batch_norm[0](xe52)
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xe42, resize(xu1, xe42.shape[2:])], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))
        xd12=self.batch_norm[1](xd12)
        
        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xe32, resize(xu2, xe32.shape[2:])], dim=1)
        xd21 = self.relu(self.d21(xu22))
        xd22 = self.relu(self.d22(xd21))
        xd22=self.batch_norm[2](xd22)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xe22, resize(xu3, xe22.shape[2:])], dim=1)
        xd31 = self.relu(self.d31(xu33))
        xd32 = self.relu(self.d32(xd31))
        xd32=self.batch_norm[3](xd32)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xe12, resize(xu4, xe12.shape[2:])], dim=1)
        xd41 = self.relu(self.d41(xu44))
        xd42 = self.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
class MDEModel(nn.Module):
    def __init__(self):
        super(MDEModel, self).__init__()
        self.upsample_mode = 'nearest'
        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.layers=[512,256,128,64]
        self.lrelu=nn.LeakyReLU()
        self.bootleneck=BottleNeckBlock(512,3)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.convOut = nn.Conv2d(64,64, 4, 2)
        self.up0=Up(1024,512)
        self.up1=Up(512,256)
        self.up2=Up(256,128)
        self.up3=Up(128,64)
        self.up = nn.Conv2d(64, 1, kernel_size=1)
        self.batch_norm=nn.ModuleList([nn.BatchNorm2d(l) for l in self.layers])
        self.pool = nn.MaxPool2d(2, stride=2)

        # self.fup1 = nn.Upsample(scale_factor=4)
        # self.fup2 =nn.Conv2d(64, 1, kernel_size=3,stride=2, padding=1, bias=False)
    
    
    def forward(self, x):
        # x=x.permute(0, 3,1, 2).float()
        # x=x.float()
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.lrelu(x)
        # x = self.encoder.maxpool(x)
        # x=self.pool(x)
        x1 = self.encoder.layer1(x)
        x1 = self.batch_norm[3](x1)
        x1 = self.lrelu(x1)
        # x11=self.pool(x1)
        # x11 = F.interpolate(x11, scale_factor=2, mode="nearest")
        x2 = self.encoder.layer2(x1)
        x2 = self.batch_norm[2](x2)
        x2 = self.lrelu(x2)
        # x22=self.pool(x2)
        # x22 = F.interpolate(x22, scale_factor=2, mode="nearest")

        x3 = self.encoder.layer3(x2)
        x3 = self.batch_norm[1](x3)
        x3 = self.lrelu(x3)
        # x33=self.pool(x3)
        # x33 = F.interpolate(x33, scale_factor=2, mode="nearest")
        
        x4 = self.encoder.layer4(x3)
        # x=self.up(x5)
        # x5=self.encoder.avgpool(x5)
        x=self.bootleneck(x4)
        
        x=self.up0(x,x4)
        
        x=self.up1(x,x3)
        x = self.batch_norm[1](x)
        x=self.up2(x,x2)
        x = self.batch_norm[2](x)
        x=self.up3(x,x1)
        x = self.batch_norm[3](x)
        # x=self.convOut(x)
        x=self.up(x)
        # x=self.fup1(x)
        # x=self.fup2(x)
        # x = F.interpolate(x, scale_factor=, mode="nearest")
        x=self.lrelu(x)
        return x
