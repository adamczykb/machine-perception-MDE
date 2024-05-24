import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    Identity
)
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
       

        self.module=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d( 4,32, kernel_size=3,stride=1,bias=False)),
            ('bnom1',  nn.BatchNorm2d(32)),

          ('relu1',  nn.GELU()),
           ('conv2',  nn.Conv2d( 32,64, kernel_size=4,stride=1,bias=False)),
            ('bnom2',  nn.BatchNorm2d(64)),
          ('gelu2',  nn.GELU()),
           ('conv3',  nn.Conv2d( 64,128, kernel_size=4,stride=2,bias=False)),
            ('bnom3',   nn.BatchNorm2d(128)),
          ('gelu3',  nn.GELU()),
           ('conv4',  nn.Conv2d( 128,256, kernel_size=4,stride=1,bias=False)),
            ('bnom4',   nn.BatchNorm2d(256)),
          ('gelu4',   nn.GELU()),
           ('conv5',  nn.Conv2d( 256,512, kernel_size=3,stride=1,bias=False)),
          ('sigmoid',  nn.Sigmoid())
        ]))



    def forward(self, x):
        y_hat = self.module(x)
        
        return y_hat