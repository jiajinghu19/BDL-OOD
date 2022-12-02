import numpy as np
import pandas as pd

import torch
import torch.optim
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import math

import DCGAN_VAE_pixel as DVAE

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ngpu = 0
nz = 100
ngf = 64
nc = 3
image_size = 32
state_G = "../LMPBT_fork/cifar100_netG.pth"
state_E = "../LMPBT_fork/cifar100_netE.pth"
batch_size = 9

transform = transforms.Compose([
        transforms.Resize((image_size)),
        transforms.ToTensor(),
    ])
dataset = dset.CIFAR100(root="./CIFAR100", train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

netG = DVAE.DCGAN_G(image_size, nz, nc, ngf, ngpu)
state_G = torch.load(state_G, map_location=device)
netG.load_state_dict(state_G, strict=False)

netE = DVAE.Encoder(image_size, nz, nc, ngf, ngpu)
state_E = torch.load(state_E, map_location=device)
netE.load_state_dict(state_E, strict=False)

netG.to(device)
netG.eval()
netE.to(device)
netE.eval()

with torch.no_grad():
    for i, (x, _) in enumerate(dataloader):
        print("x",x)
        # print("x",x.size())
        save_image(
            x,
            'x.png',
            nrow=int(math.sqrt(batch_size)),
            padding=0
        )

        x = x.to(device)
        [z,mu,logvar] = netE(x)

        # z = torch.randn(1, nz, 1, 1).to(device)
        sample = netG(z)
        # print("sample.size()",sample.size()) # sample.size() torch.Size([1, 3, 32, 32, 256])
        print("torch.max(sample,4).size()", torch.amax(sample,4).size())
        # idx = torch.LongTensor([2,1,0])
        samples =  ((torch.amax(sample,4) + 128)/255) #.index_select(1, idx)
        print("samples", samples)
        save_image(
            samples,
            'test.png',
            nrow=int(math.sqrt(batch_size)),
            padding=0
        )
        break