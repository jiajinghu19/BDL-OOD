
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import DCGAN_VAE_pixel as DVAE
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import time
from get_torchvision_dataset import get_torchvision_dataset

def KL_div(mu, logvar, reduction='none'):
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))

    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return KL


import glob, itertools
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


def store_NLL(x, recon, mu, logvar, z):
    with torch.no_grad():
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1, 256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b, -1), 1)

        log_p_z = -torch.sum(z ** 2 / 2 + np.log(2 * np.pi) / 2, 1)
        z_eps = z - mu
        z_eps = z_eps.view(opt.repeat, -1)
        log_q_z_x = -torch.sum(z_eps ** 2 / 2 + np.log(2 * np.pi) / 2 + logvar / 2, 1)

        weights = log_p_x_z + log_p_z - log_q_z_x

    return weights


def compute_NLL(weights):
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max()))) + weights.max())

    return NLL_loss


def gradient(model):
    grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
    return grad.detach()


def weights(model):
    wts = torch.cat([p.flatten() for p in model.parameters()])
    return wts


def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s + size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    return {"params": flat, "indices": indices}


def recover_flattened(flat_params, indices, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(model.parameters()):
        l[i] = l[i].view(*p.shape)
    return l


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    torch_utils.clip_grad_norm_(x, clip_value)
    g = torch.cat([p.flatten() for p in g])
    return g


def compute_likelihood(dataset, model, model_g, eig_val, eig_vec, eig_val2, eig_vec2, index, criterion):
    batch_n = 16
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_n, num_workers=1)
    model.train()
    model_g.train()
    weight = weights(model)
    temp = list()
    temp2 = list()
    count = 0
    # h_diag[h_diag == float('inf')] = 0
    for i, (x, _) in enumerate(dataloader):
        count += 1
        x = x.to(device)
        grads2_g = []
        grads_t_g = 0
        grads2_e = []
        grads_t_e = 0

        time_start = time.clock()

        for m in range(batch_n):
            ################################################# Stablizing the Gradient #####################################
            # Compute the gradient of each sample by Gradient of the batch dataset - Gradient of ( batch dataset - i-th sample)
            new_x = torch.cat((x[0:m, ], x[m + 1:, :, :]))
            target = Variable(new_x.data.view(-1) * 255).long()
            model.zero_grad()
            model_g.zero_grad()
            tmp_num_data = batch_n
            [z, mu, logvar] = model(new_x)
            recon = model_g(z)
            recon = recon.contiguous()
            recon = recon.view(-1, 256)
            recl = criterion(recon, target)
            recl = torch.sum(recl) / tmp_num_data
            kld = KL_div(mu, logvar)
            loss = recl + kld.mean()
            loss.backward()
            grads2_g.append(gradient(model_g).detach())
            grads_t_g += gradient(model_g).detach()
            model_g.zero_grad()
            grads2_e.append(gradient(model).detach())
            grads_t_e += gradient(model).detach()
            model.zero_grad()

        for m in range(batch_n):
            grads_g = grads_t_g - (batch_n - 1) * grads2_g[m]
            grads_e = grads_t_e - (batch_n - 1) * grads2_e[m]

            diff = 0

            for j in range(30):
                vec = eig_vec2[j].unsqueeze(1)
                vect = vec.t()
                grads_u = grads_e.unsqueeze(1)
                temp_val = (1 / eig_val2[j]) * torch.einsum('ij, ji, ik->ik', vec, vect, grads_u)
                temp_val = torch.einsum('ij, ij->ij', temp_val, grads_u)
                diff += temp_val.sum()

            # print(diff.sum())
            # run your code

            temp.append(diff.sum())
        time_elapsed = (time.clock() - time_start)
        print (time_elapsed)
        if count > 160:
            break

    return torch.stack(temp).cpu().detach()


if __name__ == "__main__":
    ################################################## Model arguments ##################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_distro_dataset', default='CIFAR10', help='path to in distribution dataset')
    parser.add_argument('--out_distro_dataset', default='CIFAR10', help='path to out of distribution dataset')
    parser.add_argument('--eigenvalues', default='./eigenvalues_50_vae_cifar100.npy', help='path to eigenvalues file')
    parser.add_argument('--eigenvectors', default='./eigenvectors_50_vae_cifar100.npy', help='path to eigenvectors file')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--image_size', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    parser.add_argument('--repeat', type=int, default=200)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--state_E', default='./models/cifar100_netE.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G', default='./models/cifar100_netG.pth', help='path to generator checkpoint')

    opt = parser.parse_args()

    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    transform=transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.ToTensor()
    ])
    # load test sets
    in_distro_dataset = get_torchvision_dataset(opt.in_distro_dataset, False, transform)
    out_distro_dataset = get_torchvision_dataset(opt.out_distro_dataset, False, transform)

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)

    ################################################## Model loading ##################################################################
    print('Building models...')
    netG = DVAE.DCGAN_G(opt.image_size, nz, nc, ngf, ngpu)
    state_G = torch.load(opt.state_G, map_location=device)
    netG.load_state_dict(state_G, strict=False)

    netE = DVAE.Encoder(opt.image_size, nz, nc, ngf, ngpu)
    state_E = torch.load(opt.state_E, map_location=device)
    netE.load_state_dict(state_E, strict=False)

    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    print('Building complete...')
    NLL_test_indist = []
    NLL_test_indist_bg = []

    eig_val = np.load(opt.eigenvalues)
    eig_vec = np.load(opt.eigenvectors)
    eig_val = torch.tensor(eig_val).to(device)
    eig_vec = torch.tensor(eig_vec).to(device)

    eig_val2 = np.load(opt.eigenvalues)
    eig_vec2 = np.load(opt.eigenvectors)
    eig_val2 = torch.tensor(eig_val2).to(device)
    eig_vec2 = torch.tensor(eig_vec2).to(device)

    dic_param = flatten_params(netE.parameters())

    ################################################## Get_LMPBT_score ##################################################################
    in_distro_nll = compute_likelihood(in_distro_dataset, netE, netG, eig_val, eig_vec, eig_val2, eig_vec2,
                                dic_param['indices'], loss_fn)
    out_distro_nll = compute_likelihood(out_distro_dataset , netE, netG, eig_val, eig_vec, eig_val2, eig_vec2,
                                dic_param['indices'], loss_fn)

    np.save('in_distro_nll_vae.npy', in_distro_nll.cpu().detach().numpy())
    np.save('out_distro_nll_vae.npy', out_distro_nll.cpu().detach().numpy())
