# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import calMetric as m
import calData as d
#CUDA_DEVICE = 0

start = time.time()
#loading data sets

def get_transform(dataset_name=""):
    # calData.py uses hardcoded image transforms so I'm (Harry) leaving this for now
    normalize_transform = transforms.Normalize( # default is CIFAR-10
        (125.3/255, 123.0/255, 113.9/255),
        (63.0/255, 62.1/255.0, 66.7/255.0)
    )
    # if dataset_name == "SVHN":
    #     normalize_transform = transforms.Normalize(
    #         (0.43768218, 0.44376934, 0.47280428), 
    #         (0.1980301, 0.2010157, 0.19703591)
    #     )
    return transforms.Compose([
        transforms.ToTensor(),
        normalize_transform
    ])

criterion = nn.CrossEntropyLoss()



def test(nnName, in_dataset_name, out_data_name, CUDA_DEVICE, epsilon, temperature):
    net1 = torch.load("../models/{}.pth".format(nnName))
    optimizer1 = optim.SGD(net1.parameters(), lr = 0, momentum = 0)
    net1.cuda(CUDA_DEVICE)
    
    testset_out = None
    testloader_out = None
    if out_data_name != "Uniform" and out_data_name != "Gaussian": # if the test data is not unniform or gaussian
        # TODO MNIST
        # TODO Fashion MNIST
        if out_data_name == "CIFAR-10": 
            testset_out = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=get_transform("CIFAR-10"))
        elif out_data_name == "SVHN": 
            testset_out = torchvision.datasets.SVHN(root='svhn', split='test', download=True, transform=get_transform("SVHN"))                                
        else:
            testset_out = torchvision.datasets.ImageFolder("../data/{}".format(out_data_name), transform=get_transform("CIFAR-10")) # load the data from the folder
        testloader_out = torch.utils.data.DataLoader(testset_out, batch_size=1,
                                            shuffle=False, num_workers=2)
        
    testset_in = None
    if in_dataset_name == "CIFAR-10": 
        testset_in = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=get_transform("CIFAR-10"))
    elif in_dataset_name == "CIFAR-100": 
        testset_in = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=get_transform("CIFAR-10"))
    elif in_dataset_name == "SVHN":
        testset_in = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=get_transform("SVHN"))
    else:
        print("Invalid in-distribution dataset name")
    testloader_in = torch.utils.data.DataLoader(testset_in, batch_size=1,
                                        shuffle=False, num_workers=2)
    # TODO SVHN
    # TODO MNIST
    # TODO Fashion MNIST
    
    if out_data_name == "Gaussian":
        d.testGaussian(net1, criterion, CUDA_DEVICE, testloader_in, testloader_in, nnName, out_data_name, epsilon, temperature)
        m.metric(nnName, in_dataset_name, out_data_name)

    elif out_data_name == "Uniform":
        d.testUni(net1, criterion, CUDA_DEVICE, testloader_in, testloader_in, nnName, out_data_name, epsilon, temperature)
        m.metric(nnName, in_dataset_name, out_data_name)
    else:
        d.testData(net1, criterion, CUDA_DEVICE, testloader_in, testloader_out, nnName, in_dataset_name, out_data_name, epsilon, temperature) 
        m.metric(nnName, in_dataset_name, out_data_name)








