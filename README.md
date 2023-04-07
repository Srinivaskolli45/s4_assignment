
# Transformer implementation from scratch for CIFAR10

This repository contains the scratch implementation of a simple **Transformer** architecture for the CIFAR10 dataset.

## Data

The CIFAR10 dataset was imported using `torchvision.datasets` and augmented using the [albumentations](https://albumentations.ai) library. The following augmentations were applied on the data:
* RandomCrop of 32, 32 (after padding of 4)
* FlipLR 
* CutOut(8, 8)


## Architecture


*  3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
* Apply GAP and get 1x1x48, call this X
* Create a block called ULTIMUS that:
    * Creates 3 FC layers called K, Q and V such that:
        * X*K = 48*48x8 > 8
        * X*Q = 48*48x8 > 8 
        * X*V = 48*48x8 > 8 
    * then create AM = SoftMax(Q.T * K)/(8^0.5) = 8*8 = 8
    * then Z = V*AM = 8*8 > 8
    * then another FC layer called Out such that: 
        * Z*Out = 8*8x48 > 48
* Repeat this Ultimus block 4 times
* Then add final FC layer that converts 48 to 10 and sends it to the loss function.


## Training Technique

<img width="382" alt="Screenshot 2023-02-26 at 20 16 11" src="https://user-images.githubusercontent.com/34182074/221417731-65d38992-86a3-45cd-9dd2-37250b041db4.png">
This model was trained for 24 epochs using the One-Cycle Policy with the Adam optimizer. The One Cycle Policy, introduced by Leslie Smith in 2018, is a learning rate scheduling technique that aims to achieve better accuracy in less time. It involves a cyclic increase and decrease of the learning rate during training, where the learning rate starts low, then gradually increases until it reaches a maximum value, and then decreases back to the starting value. This helps the model to converge faster and reach a better generalization performance.

The learning rate at which the loss was the lowest was taken as the `max_lr` while the `min_lr` was taken to be one-tenth of it. No annihilation was applied at the end.
