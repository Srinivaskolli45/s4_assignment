!git clone https://github.com/Srinivaskolli45/s4_assignment.git

!pip install torchsummary
!pip install torch_lr_finder


import sys
sys.path.append('/content/s4_assignment/')
sys.path.append('/content/s4_assignment/models')
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
from model9 import *
import dataloader
from torchvision import datasets
from torch_lr_finder import LRFinder
from torchsummary import summary
import seaborn as sns


SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available:", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    BATCH_SIZE=512
else:
    BATCH_SIZE=32
    

# Data
print('==> Preparing data..')

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True )
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)


train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train=True), batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset, train=False), batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=1)

# Model
print(' Building the model..')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
net = Transformer_VIT().to(device)
print(summary(net, input_size=(3, 32, 32)))

!pip install torch_lr_finder

from torch_lr_finder import LRFinder


exp_net = copy.deepcopy(net).to(device)
optimizer = torch.optim.Adam(exp_net.parameters(), lr=0.001)
criterion = nn.NLLLoss()

#define instance of LRFinder
lr_finder = LRFinder(exp_net, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, end_lr=10, num_iter=200)
lr_finder.plot()

min_loss = min(lr_finder.history['loss'])
ler_rate_1 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
print("Max LR is {}".format(ler_rate_1))

from torch.optim.lr_scheduler import OneCycleLR

ler_rate = ler_rate_1
print("Determined Max LR is:", ler_rate)


train_net_1 = copy.deepcopy(net).to(device)
optimizer = torch.optim.Adam(train_net_1.parameters(), lr=(ler_rate/10))


epochs = 24
steps_per_epoch = len(train_loader) 
total_steps = epochs * len(train_loader)
pct_start = (5*steps_per_epoch)/total_steps
print(f'pct_start --> {pct_start}')

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=ler_rate,
                                                steps_per_epoch=steps_per_epoch, 
                                                epochs=epochs,
                                                pct_start=pct_start,
                                                div_factor=10,
                                                three_phase=False, 
                                                final_div_factor=50,
                                                anneal_strategy='linear'
                                                ) 


train_net_1, history = fit_model(
    train_net_1, device=device,
    criterion = nn.CrossEntropyLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer, 
    scheduler=scheduler, 
    NUM_EPOCHS=24
)

#train accuracy plots
training_acc, training_loss, testing_acc, testing_loss, lr_trend = history

sns.lineplot(x = list(range(1, 25)), y = training_acc)
sns.lineplot(x = list(range(1, 25)), y = testing_acc)
sns.lineplot(x = list(range(1, 25)), y = training_loss)
sns.lineplot(x = list(range(1, 25)), y = testing_loss)
sns.lineplot(x = list(range(1, len(lr_trend)+1)), y = lr_trend)
