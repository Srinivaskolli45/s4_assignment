!git clone https://github.com/Srinivaskolli45/srinivas.git
%cd srinivas/
!git pull origin
%cd ..
!pip install torchsummary

'''Train CIFAR10 with PyTorch.'''
!pip install grad-cam
import sys
sys.path.append('/content/srinivas/')
sys.path.append('/content/srinivas/models')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import albumentations as Alb
import torchvision
import torchvision.transforms as transforms
import numpy as np
from utils import *
from testModel import *
from trainModel import *
from custom_resnet import *
#from utils import progress_bar
from albumentation_helper import *
import dataloader


#working on GradCAM 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 512

# Data
print('==> Preparing data..')

use_cuda = torch.cuda.is_available()

CustomTrainTransform = Alb.Compose([Alb.Normalize(mean=(0.49139968, 0.48215841, 0.44653091),std=(0.24703223, 0.24348513, 0.26158784)),
                                    Alb.PadIfNeeded(min_height=40, min_width=40, border_mode=4, value=None, p=1.0),
                                    Alb.RandomCrop(32,32),
                                    Alb.HorizontalFlip(p=0.1),
                                    Alb.Cutout(8,8,p=0.2)])
 
transform = Alb_Transforms(CustomTrainTransform)
TestTransform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])
visualTransform = transforms.Compose([transforms.ToTensor()])
#why we need normalized test data
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
trainloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=True,transform=transform,batchSize=batch_size,isShuffle=True,workers=4,needDownload=True)
testloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=False,transform=TestTransform,batchSize=batch_size,isShuffle=False,workers=4,needDownload=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#visualize the 20 images from train set
vis_dataloader = dataloader.getDataLoader(dataset=torchvision.datasets.CIFAR10, isTrain=True,transform=visualTransform,batchSize=batch_size,isShuffle=True,workers=4,needDownload=True)

# Model
print(' Building the model..')

net = custom_resnet()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#summary(net, input_size=(3, 32, 32))

!pip install torch_lr_finder

from torch_lr_finder import LRFinder

#define instance of LRFinder
criterion_new = nn.CrossEntropyLoss()
optimizer_new = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

lr_finder = LRFinder(net,optimizer_new,criterion_new)

#Run range test
lr_finder.range_test(trainloader,start_lr=1e-03,end_lr=1,num_iter=100,step_mode="exp")
lr_finder.plot()
lr_finder.reset()

from torch.optim.lr_scheduler import OneCycleLR

ler_rate = 0.04912
#Let's start with loading the trained model and train again further

#load parameters to model
#net.load_state_dict(model_parameters)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Train again with one cycle policy now
epochs = 24
steps_per_epoch = len(trainloader) 
total_steps = epochs * len(trainloader)



pct_start = (10*steps_per_epoch)/total_steps
print(f'pct_start --> {pct_start}')
scheduler = OneCycleLR(optimizer,max_lr=ler_rate,
                       steps_per_epoch=steps_per_epoch,epochs=epochs,
                       pct_start=pct_start,div_factor=10,final_div_factor=10,verbose=False)

#Train the model further
for epoch in range(start_epoch, start_epoch+24):
    train_model(dataloader=trainloader,network=net,lossfn=criterion,optimizer=optimizer,scheduler=scheduler)
    test_model(dataloader=testloader,network=net,lossfn=criterion)
