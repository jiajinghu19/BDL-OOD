
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import DCGAN_VAE_pixel as DVAE
import torch.nn.functional as F
from pathlib import Path
import os
from get_torchvision_dataset import get_torchvision_dataset

def KL_div(mu,logvar,reduction = 'avg'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1) 
        return KL


def perturb(x, mu,device):
    b,c,h,w = x.size()
    mask = torch.rand(b,c,h,w)<mu
    mask = mask.float().to(device)
    noise = torch.FloatTensor(x.size()).random_(0, 256).to(device)
    x = x*255
    perturbed_x = ((1-mask)*x + mask*noise)/255.
    return perturbed_x


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--dataset', default='CIFAR10', help='dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--image_size', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help = 'hidden channel sieze')
    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=1., help='beta for beta-vae')

    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--perturbed', action='store_true', help='Whether to train on perturbed data, used for comparing with likelihood ratio by Ren et al.')
    parser.add_argument('--ratio', type=float, default=0.2, help='ratio for perturbation of data, see Ren et al.')

    opt = parser.parse_args()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if opt.experiment is None:
        opt.experiment = './models/{}'.format(opt.dataset)
    Path(opt.experiment).mkdir(parents=True, exist_ok=True)
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((opt.image_size)),
        transforms.ToTensor(),
    ])
    dataset = get_torchvision_dataset(opt.dataset, True, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = DVAE.DCGAN_G(opt.image_size, nz, nc, ngf, ngpu)
    netG.apply(weights_init)
    # print (len(netG.parameters))
    netE = DVAE.Encoder(opt.image_size, nz, nc, ngf, ngpu)
    netE.apply(weights_init)

    
    netE.to(device)
    netG.to(device)
    # setup optimizer
    
    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, weight_decay = 3e-5)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay = 3e-5)

    netE.train()
    netG.train()

    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    rec_l = []
    kl = []
    tloss = []
    best_rec_l = None
    best_rec_l_epoch = None
    best_rec_l_netG = None
    best_rec_l_netE = None
    for epoch in range(opt.niter):
        epoch_rec_l = []
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            print("x",x)
            print("x.size()",x.size())
            if opt.perturbed:
                x = perturb(x, opt.ratio, device)

            b = x.size(0)
            print("x.data.view(-1)",x.data.view(-1).size())
            target = Variable(x.data.view(-1) * 255).long()
            [z,mu,logvar] = netE(x)
            recon = netG(z)
            print("recon.size() 1",recon.size())
            recon = recon.contiguous()
            print("recon.size() 2",recon.size())
            recon = recon.view(-1,256)
            print("recon.size() 3",recon.size())
            print("target.size()",target.size())
            print("recon",recon)
            print("target",target)
            recl = loss_fn(recon, target)
            recl = torch.sum(recl) / b
            kld = KL_div(mu,logvar)
            
            loss =  recl + opt.beta*kld.mean()
            
                
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss = loss
            loss.backward(retain_graph=True)
            

            optimizer1.step()
            optimizer2.step()
            epoch_rec_l.append(recl.detach().item())
            rec_l.append(recl.detach().item())
            kl.append(kld.mean().detach().item())
            tloss.append(loss.detach().item())
           
            if not i % 100:
                print('epoch:{} recon:{} kl:{}'.format(epoch,np.mean(rec_l),np.mean(kl)
                    ))
        
        if (best_rec_l is None) or np.mean(epoch_rec_l) < best_rec_l:
            best_rec_l = np.mean(epoch_rec_l)
            best_rec_l_epoch = epoch
            torch.save(netG.state_dict(), os.path.join(opt.experiment,'{}_netG.pth'.format(opt.dataset)))
            torch.save(netE.state_dict(), os.path.join(opt.experiment,'{}_netE.pth'.format(opt.dataset)))
    torch.save(netG.state_dict(), os.path.join(opt.experiment,'{}_epoch{}_netG.pth'.format(opt.dataset,epoch)))
    torch.save(netE.state_dict(), os.path.join(opt.experiment,'{}_epoch{}_netE.pth'.format(opt.dataset,epoch)))
    print("Done training!")
    print("Best epoch was {}".format(best_rec_l_epoch))
