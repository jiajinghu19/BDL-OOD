#!/usr/bin/env python3

import argparse
import torch

import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import os
import shutil

import densenet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=('cifar10', 'svhn', 'FashionMNIST'))
    parser.add_argument('--early_stop', type=int, default=20) # stop training early if the model starts overfitting
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)


    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    if args.dataset == "svhn": # calculated using `compute_dataset_means_stds.py`
        normMean = [ 0.43768218, 0.44376934, 0.47280428 ] 
        normStd = [ 0.1980301, 0.2010157, 0.19703591 ]
    if args.dataset == "FashionMNIST": # calculated using `compute_dataset_means_stds.py`
        normMean = [ 0.2860402 ] 
        normStd = [ 0.3530239 ]

    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normTransform
    ])

    train_dataset = None
    test_dataset = None
    input_channels = 3
    if args.dataset == "svhn":
        train_dataset = dset.SVHN(root='svhn', split='train', download=True,
                    transform=trainTransform)
        test_dataset = dset.SVHN(root='svhn', split='test', download=True,
                    transform=testTransform)
    if args.dataset == "FashionMNIST":
        train_dataset = dset.FashionMNIST(root='FashionMNIST', train=True, download=True,
                    transform=trainTransform)
        test_dataset = dset.FashionMNIST(root='FashionMNIST', train=False, download=True,
                    transform=testTransform)
        input_channels = 1
    else: #cifar10
        train_dataset = dset.CIFAR10(root='cifar', train=True, download=True,
                     transform=trainTransform)
        test_dataset = dset.CIFAR10(root='cifar', train=False, download=True,
                     transform=testTransform)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        train_dataset,
        batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        test_dataset,
        batch_size=args.batchSz, shuffle=False, **kwargs)

    net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10,input_channels=input_channels)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    best_test_error = None
    best_test_epoch = None
    early_stopping_counter = 0
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test_error = test(args, epoch, net, testLoader, optimizer, testF)
        if (best_test_error is None) or (test_error<best_test_error): # if this is the new best test error
            best_test_error = test_error # record the new best test error
            best_test_epoch = epoch # record the epoch at which this was saved
            early_stopping_counter = 0 # reset the early stopping counter
            torch.save(net, os.path.join(args.save, 'best_test_error.pth'))
            print("SAVED BEST TEST ERROR. Epoch ", best_test_epoch, ", Error ", test_error)
        else:
            early_stopping_counter = early_stopping_counter + 1 # increment the early stopping counter
            torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

        if early_stopping_counter >= args.early_stop: # if we should stop early
            print("Stopping early at epoch {}. Best saved model at epoch {}. Best test error {}".format(
                epoch,best_test_epoch, best_test_error
            ))
            break

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data.item(), err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data.item(), err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
