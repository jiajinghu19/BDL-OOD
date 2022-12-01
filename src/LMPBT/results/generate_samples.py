import numpy as np
import pandas as pd

import torch
import torch.optim
import matplotlib.pyplot as plt

from torchvision.utils import save_image

import DCGAN_VAE_pixel as DVAE

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ngpu = 0
nz = 100
ngf = 64
nc = 3
image_size = 32
state_G = "../LMPBT_fork/cifar100_netG.pth"

netG = DVAE.DCGAN_G(image_size, nz, nc, ngf, ngpu)
state_G = torch.load(state_G, map_location=device)
netG.load_state_dict(state_G, strict=False)

# netE = DVAE.Encoder(image_size, nz, nc, ngf, ngpu)
# state_E = torch.load(state_E, map_location=device)
# netE.load_state_dict(state_E, strict=False)

netG.to(device)
netG.eval()
# netE.to(device)
# netE.eval()

with torch.no_grad():
    sample = torch.randn(1, nz, 1, 1).to(device) # this is correct
    sample = netG(sample)
    print("sample.size()",sample.size())
    print("torch.movedim(sample,4,0).size()",torch.squeeze(torch.movedim(sample,4,0)).size())
    save_image(
        torch.squeeze(torch.movedim(sample,4,0)),
        'test.png',
        nrow=32, 
        padding=0
    )