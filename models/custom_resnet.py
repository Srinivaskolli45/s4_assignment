# Custom resnet model
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#ResNet architecture for CIFAR10
class custom_resnet(nn.Module):
  def __init__(self):
    super(custom_resnet, self).__init__()

    self.prepLayer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
      )
    self.Layer1_X = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(128),
        nn.ReLU()
      )
    self.Layer1_R1 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128)
      )
    self.Layer2 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1, bias=False),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(256),
        nn.ReLU()
      )
    self.Layer3_X = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(512),
        nn.ReLU()
      )
    self.Layer3_R2 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU()
      )
    self.maxpool = nn.MaxPool2d(4,4)
    self.fc = nn.Linear(512, 10)

  def forward(self, x):
      x = self.prepLayer(x)

      x = self.Layer1_X(x)
      r1= self.Layer1_R1(x)
      x = x + r1

      x = self.Layer2(x)

      x = self.Layer3_X(x)
      r2 =self.Layer3_R2(x)
      x =  x +r2

      x = self.maxpool(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)
      return F.log_softmax(x, dim=1)  

