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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 =nn.Conv2d(in_channels, in_channels // 2, kernel_size=2,stride=2,  bias=False)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = self.up1(x1)
        # x1 = self.up2(x1)
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
      
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class BottleNeckBlock(nn.Module):
    def __init__(
        self, in_channels, kernel_size=3, padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = nn.Conv2d(in_channels,in_channels, kernel_size, strides)
        self.convB = nn.Conv2d(in_channels*2,in_channels, kernel_size, strides)
        self.reluA = nn.LeakyReLU()
        self.reluB = nn.LeakyReLU()

    def forward(self, xin):
        x = self.convA(xin)
        x = self.reluA(x)

        diffY = x.size()[2] - xin.size()[2]
        diffX = x.size()[3] - xin.size()[3]
      
        
        xin = F.pad(xin, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        dec4 = torch.cat((x, xin), dim=1)
        x = self.convB(dec4)
        # x = self.convB(x)
        x = self.reluB(x)
        return x


class MDEModel(nn.Module):
    def __init__(self):
        super(MDEModel, self).__init__()
        self.upsample_mode = 'nearest'
        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.layers=[512,256,128,64]
        self.lrelu=nn.LeakyReLU()
        self.bootleneck=BottleNeckBlock(512,2)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.convOut = nn.Conv2d(1,1, 1, 1)
        self.up0=Up(512,512)
        self.up1=Up(512,256)
        self.up2=Up(256,128)
        self.up3=Up(128,64)
        self.up = nn.ConvTranspose2d(64, 1, kernel_size=8, stride=2)
        self.batch_norm=nn.ModuleList([nn.BatchNorm2d(l) for l in self.layers])
        self.fup1 = nn.Upsample(scale_factor=4)
        self.fup2 =nn.Conv2d(64, 1, kernel_size=3,stride=2, padding=1, bias=False)
    
    
    def forward(self, x):
        # x=x.permute(0, 3,1, 2).float()
        # x=x.float()
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.lrelu(x)
        # x = self.encoder.maxpool(x)

        x1 = self.encoder.layer1(x)
        x1 = self.batch_norm[3](x1)
        # x1 = self.gelu(x1)

        x2 = self.encoder.layer2(x1)
        x2 = self.batch_norm[2](x2)
        # x2 = self.gelu(x2)

        x3 = self.encoder.layer3(x2)
        x3 = self.batch_norm[1](x3)
        # x3 = self.gelu(x3)

        x4 = self.encoder.layer4(x3)

        # x=self.up(x5)
        # x5=self.encoder.avgpool(x5)
        x=self.bootleneck(x4)
        x=self.up1(x,x3)
        x = self.batch_norm[1](x)
        x=self.up2(x,x2)
        x = self.batch_norm[2](x)
        x=self.up3(x,x1)
        x = self.batch_norm[3](x)
        x=self.fup1(x)
        x=self.fup2(x)
        x=self.convOut(x)
        # x = F.interpolate(x, scale_factor=, mode="nearest")
        # x=F.tanh(x)
        return x
