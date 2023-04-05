from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self,norm_type = "nn.BatchNorm2d"):
        super(Net, self).__init__()
        
        if norm_type == 'nn.BatchNorm2d':
            self.Batch_norm_func()
        elif norm_type == 'nn.GroupNorm':
            self.Group_norm_func()
        elif norm_type == 'nn.LayerNorm':
            self.Layer_norm_func()

    def Group_norm_func(self):
        #input 1x28x28   ==> Output 10x26x26  | RF 3
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, 3),
                                      nn.GroupNorm(2,10),
                                      nn.ReLU())
                                      
        #input 10x26x26 ==> Output 10x24x24 | RF 5
        self.conv2 = nn.Sequential(nn.Conv2d(10, 10, 3),
                                      nn.GroupNorm(2,10),
                                      nn.ReLU())
                                      
        #input 10x24x24 ==> Output 10x12x12 | RF  7
        self.transitionblk1 = nn.MaxPool2d(2,2)

        #input 10x12x12 ==> Output 32x10x10 | RF 10
        self.conv3 = nn.Sequential(nn.Conv2d(10,32,3),
                                      nn.GroupNorm(2,32),
                                      nn.ReLU())
                                  

        #input 32x10x10 ==> Output 10x8x8 | RF 14
        self.conv4 = nn.Sequential(nn.Conv2d(32,10,3),
                                      nn.GroupNorm(2,10),
                                      nn.ReLU())                                 
                                              
        #input 10x8x8 ==> Output 20x6x6 | RF 18
        self.conv5 = nn.Sequential(nn.Conv2d(10,20,3),
                                      nn.GroupNorm(2,20),
                                      nn.ReLU())
                                      
        #input 20x6x6 ==> OUtput 10x6x6 | RF 20
        self.conv6 = nn.Sequential(nn.Conv2d(20,10,1),
                                    nn.GroupNorm(2,10),
                                     nn.ReLU())
                                     
        self.avgPoolblk = nn.AvgPool2d(6,6)

    def Layer_norm_func(self):
        #input 1x28x28   ==> Output 10x26x26  | RF 3
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, 3),
                                      nn.LayerNorm(10,26,26),
                                      nn.ReLU())
                                      
        #input 10x26x26 ==> Output 10x24x24 | RF 5
        self.conv2 = nn.Sequential(nn.Conv2d(10, 10, 3),
                                      nn.LayerNorm(10,24,24),
                                      nn.ReLU())
                                      
        #input 10x24x24 ==> Output 10x12x12 | RF  7
        self.transitionblk1 = nn.MaxPool2d(2,2)

        #input 10x12x12 ==> Output 32x10x10 | RF 10
        self.conv3 = nn.Sequential(nn.Conv2d(10,32,3),
                                      nn.LayerNorm(32,10,10),
                                      nn.ReLU())
                                     

        #input 32x10x10 ==> Output 10x8x8 | RF 14
        self.conv4 = nn.Sequential(nn.Conv2d(32,10,3),
                                      nn.LayerNorm(10,8,8),
                                      nn.ReLU())
                                              
        #input 10x8x8 ==> Output 20x6x6 | RF 18
        self.conv5 = nn.Sequential(nn.Conv2d(10,20,3),
                                      nn.LayerNorm(20,6,6),
                                      nn.ReLU())
                                      
        #input 20x6x6 ==> OUtput 10x6x6 | RF 20
        self.conv6 = nn.Sequential(nn.Conv2d(20,10,1),
                                    nn.LayerNorm(10,6,6),
                                     nn.ReLU())
                                     
       
        
        self.avgPoolblk = nn.AvgPool2d(6,6)

    def Batch_norm_func(self):
        #input 1x28x28   ==> Output 10x26x26  | RF 3
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, 3),
                                      nn.BatchNorm2d(10),
                                      nn.ReLU())
                                      
        #input 10x26x26 ==> Output 10x24x24 | RF 5
        self.conv2 = nn.Sequential(nn.Conv2d(10, 10, 3),
                                      nn.BatchNorm2d(10),
                                      nn.ReLU())
                                      
        #input 10x24x24 ==> Output 10x12x12 | RF  7
        self.transitionblk1 = nn.MaxPool2d(2,2)

        #input 10x12x12 ==> Output 32x10x10 | RF 10
        self.conv3 = nn.Sequential(nn.Conv2d(10,32,3),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU())
                                     

        #input 32x10x10 ==> Output 10x8x8 | RF 14
        self.conv4 = nn.Sequential(nn.Conv2d(32,10,3),
                                      nn.BatchNorm2d(10),
                                      nn.ReLU())
                                      
         
                                      
        #input 10x8x8 ==> Output 20x6x6 | RF 18
        self.conv5 = nn.Sequential(nn.Conv2d(10,20,3),
                                      nn.BatchNorm2d(20),
                                      nn.ReLU())
                                      
        #input 20x6x6 ==> OUtput 10x6x6 | RF 20
        self.conv6 = nn.Sequential(nn.Conv2d(20,10,1),
                                    nn.BatchNorm2d(10),
                                     nn.ReLU())
                                     
       
        
        self.avgPoolblk = nn.AvgPool2d(6,6)


    def forward(self, x):
        x = self.conv1(x)  #input 1x28x28   ==> Output 10x26x26  | RF 3
        x = self.conv2(x)  #input 10x26x26   ==> Output 10x24x24  | RF 5      
        x = self.transitionblk1(x) #input 10x24x24 ==> Output 10x12x12 | RF  6
        x = self.conv3(x)  #input 10x12x12 ==> Output 32x10x10 | RF 10
        x = self.conv4(x)   #input 32x10x10 ==> Output 10x8x8 | RF 14
        x = self.conv5(x)   #input 10x8x8 ==> Output 20x6x6 | RF 18
        x = self.conv6(x)        #input 20x6x6 ==> Output 10x6x6 | RF 18
        x = self.avgPoolblk(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)  



