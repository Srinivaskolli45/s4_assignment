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
    self.fc_k = nn.Linear(48, 8)
    self.fc_q = nn.Linear(48, 8)
    self.fc_v = nn.Linear(48, 8)
    self.out = nn.Linear(8, 48)

  def forward(self, x):
    
    k = self.fc_k(x)
    q = self.fc_q(x)
    v = self.fc_v(x)
    AM = F.softmax(torch.matmul(q, torch.transpose(k, 0, 1)), dim=1) / np.sqrt(8)
    z = torch.matmul(AM, v)

    out = self.out(z)
    
    return x + out


class Transformer(nn.Module):
  def __init__(self):
    super(Transformer, self).__init__()

    self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)      # Output - 16x32x32
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)      # Output - 32x32x32
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 48, 3, padding=1, bias=False)      # Output - 48x32x32

    self.gap = nn.AvgPool2d(32)            # Output - 48x1x1

    self.ultimus_prime = nn.ModuleList([Ultimus_Block() for i in range(4)])

    self.fc = nn.Linear(48, 10)

  def forward(self, x):

    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.gap(self.conv3(x))
    x = x.view(-1, 48)

    for i in range(4):
      x = self.ultimus_prime[i](x)

    x = self.fc(x)
    return F.log_softmax(x, dim=-1)

