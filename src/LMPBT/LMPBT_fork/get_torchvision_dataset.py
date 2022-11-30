
import torchvision.datasets as dset

def get_torchvision_dataset(name, train, transform):
    if name == "MNIST":
        return dset.MNIST(root="./MNIST", train=train, download=True, transform=transform)
    elif name == "FashionMNIST":
        return dset.FashionMNIST(root="./FashionMNIST", train=train, download=True, transform=transform)
    elif name == "SVHN":
        return dset.SVHN(root="./SVHN", split='train' if train else "test", download=True, transform=transform)
    elif name == "CIFAR10":
        return dset.CIFAR10(root="./CIFAR10", train=train, download=True, transform=transform)
    elif name == "CIFAR100":
        return dset.CIFAR100(root="./CIFAR100", train=train, download=True, transform=transform)
    
    raise ValueError("dataset name is invalid, received {}".format(name))