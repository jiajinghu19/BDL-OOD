from torchvision import datasets, transforms
import torch
import numpy as np


def load_data(train_dataset, test_dataset, train_batch_size, eval_batch_size, S):
    if train_dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    '../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.CenterCrop((S)),
                        transforms.ToTensor(), torch.round])),    
                batch_size=train_batch_size, shuffle=True)

        train_eval_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=True,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),    
        batch_size=eval_batch_size, shuffle=False)

        print("MNIST train data : %d binary images with raw shape (%d,%d)." % (
        train_loader.dataset.data.shape[0],
        train_loader.dataset.data.shape[1],
        train_loader.dataset.data.shape[2]))
        print("Requested batch_size %d, so each epoch consists of %d updates" % (
            train_batch_size,
            int(np.ceil(train_loader.dataset.data.shape[0] / train_batch_size))))
    elif train_dataset == "FashionMNIST":
        train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(
                    '../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.CenterCrop((S)),
                        transforms.ToTensor(), torch.round])),    
                batch_size=train_batch_size, shuffle=True)

        train_eval_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            '../data', train=True,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),    
        batch_size=eval_batch_size, shuffle=False)
    elif train_dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    '../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.CenterCrop((S)),
                        transforms.ToTensor(), torch.round])),    
                batch_size=train_batch_size, shuffle=True)

        train_eval_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            '../data', train=True,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),    
        batch_size=eval_batch_size, shuffle=False)
    elif train_dataset == "SVHN":
        train_loader = torch.utils.data.DataLoader(
                datasets.SVHN(
                    '../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.CenterCrop((S)),
                        transforms.ToTensor(), torch.round])),    
                batch_size=train_batch_size, shuffle=True)

        train_eval_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            '../data', train=True,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),    
        batch_size=eval_batch_size, shuffle=False)

    
    if test_dataset == "MNIST":
        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=False,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),
        batch_size=eval_batch_size, shuffle=False)
    elif test_dataset == "FashionMNIST":
        test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            '../data', train=False,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),
        batch_size=eval_batch_size, shuffle=False)
    elif test_dataset == "CIFAR10":
        test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            '../data', train=False,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),
        batch_size=eval_batch_size, shuffle=False)   
    elif test_dataset == "SVHN":
        test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            '../data', train=False,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),
        batch_size=eval_batch_size, shuffle=False) 

    return train_loader, train_eval_loader, test_loader
    
     

        