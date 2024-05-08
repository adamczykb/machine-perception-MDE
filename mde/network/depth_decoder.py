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
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
      
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
     
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MDEModel(nn.Module):
    def __init__(self):
        super(MDEModel, self).__init__()
        self.upsample_mode = 'nearest'
        self.encoder = torchvision.models.resnet34(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.up1=Up(512,256)
        self.up2=Up(256,128)
        self.up3=Up(128,64)
        self.up = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)

    
    
    def forward(self, x):
        # x=x.permute(0, 3,1, 2).float()
        x=x.float()
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x1 = self.encoder.maxpool(x)

        x2 = self.encoder.layer1(x1)
        # x2 = F.interpolate(x2, scale_factor=2, mode="nearest")
        x3 = self.encoder.layer2(x2)
        # x3 = F.interpolate(x3, scale_factor=2, mode="nearest")
        x4 = self.encoder.layer3(x3)
        # x4 = F.interpolate(x4, scale_factor=2, mode="nearest")
        x5 = self.encoder.layer4(x4)
        # x = F.interpolate(x5, scale_factor=2, mode="nearest")
        # x=self.up(x5)
        x5=self.encoder.avgpool(x5)
        x=self.up1(x5,x4)
        x=self.up2(x,x3)
        x=self.up3(x,x2)
        x=self.up(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x
