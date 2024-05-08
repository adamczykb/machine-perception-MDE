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
        seq_list=[]
        
        seq_list.append(Conv2d( 4,8, kernel_size=3,stride=1,bias=False))
        # seq_list.append(BatchNorm2d(8))
        seq_list.append(nn.GELU())
        for i in range(1,8):
            if i!=7:
                seq_list.append(Conv2d( i*8,(i+1)*8, kernel_size=3,stride=1,bias=False))
                seq_list.append(BatchNorm2d((i+1)*8))
                seq_list.append(nn.GELU())
            else:
                seq_list.append(Conv2d( i*8,i*8, kernel_size=3,stride=1,bias=False))
                seq_list.append(nn.Sigmoid())

        self.module = nn.Sequential(*seq_list)

    def forward(self, x):
        y_hat = self.module(x)
        return y_hat