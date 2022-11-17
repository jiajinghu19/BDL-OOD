#!/usr/bin/env python3

import torch

import torchvision.datasets as dset
import torchvision.transforms as transforms

def compute_dataset_means_stds(data,name=""):
    imgs = [item[0] for item in data] # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print(name,"Means: [",mean_r,mean_g,mean_b,"]")

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print(name,"Stds: [",std_r,std_g,std_b,"]")

cifar10 = dset.CIFAR10(root='cifar', train=True, download=True,
                    transform=transforms.ToTensor())
svhn = dset.SVHN(root='svhn', split='train', download=True,
                    transform=transforms.ToTensor())

compute_dataset_means_stds(cifar10,"CIFAR-10")
compute_dataset_means_stds(svhn,"SVHN")