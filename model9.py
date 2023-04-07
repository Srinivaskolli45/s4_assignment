import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Ultimus_Block(nn.Module):
  def __init__(self):
    super(Ultimus_Block, self).__init__()
    self.k = nn.Linear(in_features=48, out_features=8)
    self.q = nn.Linear(in_features=48, out_features=8)
    self.v = nn.Linear(in_features=48, out_features=8)
    self.z_out = nn.Linear(in_features=8, out_features=48)

  def forward(self, x):
    
    x_k = self.k(x)
    x_q = self.q(x)
    x_v = self.v(x)
    AM = F.softmax(torch.matmul(x_k,torch.transpose(x_q, 0, 1)), dim=1) / np.sqrt(8)
    z = torch.matmul(AM, x_v)

    out = self.z_out(z)
    
    return x + out



class Transformer_VIT(nn.Module):
  def __init__(self):
    super(Transformer_VIT, self).__init__()

    self.ultimus_blk = Ultimus_Block()

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels =3, out_channels=16, kernel_size=(3,3), padding=1, bias=False),      # Output - 16x32x32
        nn.BatchNorm2d(16),
        nn.ReLU())
    
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels =16, out_channels=32, kernel_size=(3,3), padding=1, bias=False),      # Output - 32x32x32
        nn.BatchNorm2d(32),
        nn.ReLU())
    
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels =32, out_channels=48, kernel_size=(3,3), padding=1, bias=False),      # Output - 48x32x32
        nn.BatchNorm2d(48),
        nn.ReLU())
    
    self.gap = nn.AvgPool2d(32)            # Output - 48x1x1

    self.ultimus_prime = nn.ModuleList([Ultimus_Block() for i in range(4)])

    self.fc = nn.Linear(in_features=48, out_features=10)

  def forward(self, x):

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.gap(self.conv3(x))
    x = x.view(-1, 48)

    x = self.ultimus_blk(x)
    x = self.ultimus_blk(x)
    x = self.ultimus_blk(x)
    x = self.ultimus_blk(x)
    
    x = self.fc(x)
    return F.log_softmax(x, dim=-1)
