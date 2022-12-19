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

# PARAMETERS
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ngpu = 0
nz = 100
ngf = 64
nc = 3
image_size = 32
state_G = "../LMPBT_fork/cifar100_netG.pth"
state_E = "../LMPBT_fork/cifar100_netE.pth"
batch_size = 9

# LOAD AND TRANSFORM DATA
transform = transforms.Compose([
        transforms.Resize((image_size)),
        transforms.ToTensor(),
    ])
dataset = dset.CIFAR100(root="./CIFAR100", train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# LOAD MODELS
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
    for i, (x, _) in enumerate(dataloader): # for each batch
        # save the input images
        save_image(
            x,
            'x.png',
            nrow=int(math.sqrt(batch_size)),
            padding=0
        )
        # print("x",x)


        x = x.to(device) 
        [z,mu,logvar] = netE(x) # pass the input images through the encoder

        generated = netG(z) # pass the encoding through the generator
        print("generated.size()",generated[:,:,:,:,0].size())
        generated =  (torch.amax(generated,4)) # not sure how to handle the 256 dimension, here we pull the max
        # generated = generated[:,:,:,:,0]
        # generated =  -1.0*(generated) 

        # print("generated",generated)

        # save the generated outputs
        save_image(
            generated,
            'test.png',
            nrow=int(math.sqrt(batch_size)),
            padding=0
        )
        break