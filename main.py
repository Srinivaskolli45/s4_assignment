import torch
import torchvision
import numpy as np
import copy
import torch.nn as nn
import albumentations as A
from torchvision import datasets
from train import fit_model
from data import AlbumentationImageDataset
from Assignment_9_model import Transformer
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


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True )
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)


train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train=True), batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset, train=False), batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
net = Transformer().to(device)
print(summary(net, input_size=(3, 32, 32)))


exp_net = copy.deepcopy(net).to(device)
optimizer = torch.optim.Adam(exp_net.parameters(), lr=0.001)
criterion = nn.NLLLoss()
lr_finder = LRFinder(exp_net, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, end_lr=10, num_iter=200)
lr_finder.plot()
min_loss = min(lr_finder.history['loss'])
ler_rate_1 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
print("Max LR is {}".format(ler_rate_1))

exp_net = copy.deepcopy(net).to(device)
optimizer = torch.optim.Adam(exp_net.parameters(), lr=ler_rate_1/10)
criterion = nn.NLLLoss()
lr_finder = LRFinder(exp_net, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, end_lr=ler_rate_1*10, num_iter=200)
lr_finder.plot()
min_loss = min(lr_finder.history['loss'])
ler_rate_2 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
print("Max LR is {}".format(ler_rate_2))


ler_rate = ler_rate_2
print("Determined Max LR is:", ler_rate)


train_net_1 = copy.deepcopy(net).to(device)
optimizer = torch.optim.Adam(train_net_1.parameters(), lr=(ler_rate/10))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=ler_rate,
                                                steps_per_epoch=len(train_loader), 
                                                epochs=24,
                                                pct_start=0.2,
                                                div_factor=10,
                                                three_phase=False, 
                                                final_div_factor=50,
                                                anneal_strategy='linear'
                                                ) #final_div_factor=100,
train_net_1, history = fit_model(
    train_net_1, device=device,
    criterion = nn.CrossEntropyLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer, 
    scheduler=scheduler, 
    NUM_EPOCHS=24
)


training_acc, training_loss, testing_acc, testing_loss, lr_trend = history

sns.lineplot(x = list(range(1, 25)), y = training_acc)
sns.lineplot(x = list(range(1, 25)), y = testing_acc)
sns.lineplot(x = list(range(1, 25)), y = training_loss)
sns.lineplot(x = list(range(1, 25)), y = testing_loss)
sns.lineplot(x = list(range(1, len(lr_trend)+1)), y = lr_trend)
