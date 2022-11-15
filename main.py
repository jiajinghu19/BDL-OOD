import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim
import matplotlib.pyplot as plt

from collections import OrderedDict
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from src.VAE.vae import VariationalAutoencoder
from utils.load_data import load_data

def eval_model_on_data(
        model, dataset_nickname, data_loader, device, args):
    ''' Evaluate an encoder/decoder model on a dataset

    Returns
    -------
    vi_loss : float
    bce_loss : float
    l1_loss : float
    '''
    model.eval()
    total_vi_loss = 0.0
    total_l1 = 0.0
    total_bce = 0.0
    n_seen = 0
    total_1pix = 0.0
    for batch_idx, (batch_data, _) in enumerate(data_loader):
        batch_x_ND = batch_data.to(device).view(-1, model.n_dims_data)
        total_1pix += torch.sum(batch_x_ND)
        _, loss, _ = model.calc_vi_loss(batch_x_ND, n_mc_samples=args.n_mc_samples)
        total_vi_loss += loss.item()

        # Use deterministic reconstruction to evaluate bce and l1 terms
        batch_xproba_ND = model.decode(model.encode(batch_x_ND))
        total_l1 += torch.sum(torch.abs(batch_x_ND - batch_xproba_ND))
        total_bce += F.binary_cross_entropy(batch_xproba_ND, batch_x_ND, reduction='sum')
        n_seen += batch_x_ND.shape[0]
        break 
    msg = "%s data: %d images. Total pixels on: %d. Frac pixels on: %.3f" % (
        dataset_nickname, n_seen, total_1pix, total_1pix / float(n_seen*784))

    vi_loss_per_pixel = total_vi_loss / float(n_seen * model.n_dims_data)
    l1_per_pixel = total_l1 / float(n_seen * model.n_dims_data)
    bce_per_pixel = total_bce / float(n_seen * model.n_dims_data) 
    return float(vi_loss_per_pixel), float(l1_per_pixel), float(bce_per_pixel), msg


def predict_on_data(model, data_loader, args):
    model.eval()
    probas = []
    if args.method == "VAE":
        for batch_idx, (batch_data, _) in enumerate(data_loader):
            batch_x_ND = batch_data.to(args.device).view(-1, model.n_dims_data)
            loss, _, _ = model.calc_vi_loss(batch_x_ND, n_mc_samples=args.n_mc_samples)
            probas.extend(loss.tolist())
    return probas
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OOD detection on image classification tasks')
    parser.add_argument(
        '--method', type=str, choices=["ODIN", "VAE"],
        help="which method to use, ODIN or VAE")
    parser.add_argument(
        '--train_data', type=str, choices=["MNIST", "FashionMNIST", "CIFAR10", "SVHN"],
        help="which data to use in training")
    parser.add_argument(
        '--n_epochs', type=int, default=10,
        help="number of epochs (default: 10)")
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='batch size (default: 1024)')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate for grad. descent (default: 0.001)')
    parser.add_argument(
        '--hidden_layer_sizes', type=str, default='512',
        help='Comma-separated list of size values (default: "512")')
    parser.add_argument(
        '--filename_prefix', type=str, default='$method-arch=$hidden_layer_sizes-lr=$lr')
    parser.add_argument(
        '--q_sigma', type=float, default=0.1,
        help='Fixed variance of approximate posterior (default: 0.1)')
    parser.add_argument(
       '--n_mc_samples', type=int, default=1,
       help='Number of Monte Carlo samples (default: 1)')
    parser.add_argument(  
        '--seed', type=int, default=8675309,
        help='random seed (default: 8675309)')
    parser.add_argument(
        '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', 
        help='cuda:[d] | cpu')
    args = parser.parse_args()
    args.hidden_layer_sizes = [int(s) for s in args.hidden_layer_sizes.split(',')]

    ## Set random seed
    torch.manual_seed(args.seed)
    device = args.device

    ## Set filename_prefix for results
    for key, val in args.__dict__.items():
        args.filename_prefix = args.filename_prefix.replace('$' + key, str(val))
    print("Saving with prefix: %s" % args.filename_prefix)

    if args.train_data in ["MNIST", "FashionMNIST"]:
        S = 28 
    else:
        S = 32
    n_dims_data = S**2


    ## Create AE model by calling its constructor
    if args.method == 'ODIN':
        # TODO: please add ODIN
        pass
    elif args.method == 'VAE':
        model = VariationalAutoencoder(
            n_dims_data=n_dims_data,
            q_sigma=args.q_sigma,
            hidden_layer_sizes=args.hidden_layer_sizes).to(device)
    else:
        raise ValueError("Method must be 'ODIN' or 'VAE'")

    eval_batch_size = 20000

    ## Create generators for grabbing batches of train or test data
    # Each loader will produce **binary** data arrays (using transforms defined below)
    train_loader, train_eval_loader, test_loader_in, test_loader_out = load_data(train_dataset=args.train_data,
    train_batch_size=args.batch_size, eval_batch_size=eval_batch_size, S=S)


    ## Create an optimizer linked to the model parameters
    # Given gradients computed by pytorch, this optimizer handle update steps to params
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## Training loop that repeats for each epoch:
    #  -- perform minibatch training updates (one epoch = one pass thru dataset)
    #  -- for latest model, compute performance metrics on training set
    #  -- for latest model, compute performance metrics on test set
    for epoch in range(args.n_epochs + 1):
        if epoch > 0:
            model.train_for_one_epoch_of_gradient_update_steps(
                optimizer, train_loader, device, epoch, args)

        ## Only save results for epochs 0,1,2,3,4,5 and 10,20,30,...
        if epoch > 4 and epoch % 10 != 0:
            continue

        print('==== evaluation after epoch %d' % (epoch))
        ## For evaluation, need to use a 'VAE' for some loss functions
        # This chunk will copy the encoder/decoder parameters
        # from our latest AE model into a VAE
        if args.method == 'VAE':
            tmp_vae_model = model
        else:
            # TODO: please add ODIN here
            pass

        ## Compute VI loss (bce + kl), bce alone, and l1 alone
        tr_loss, tr_l1, tr_bce, tr_msg = eval_model_on_data(
            tmp_vae_model, 'train', train_eval_loader, device, args)
        if epoch == 0:
            print(tr_msg) # descriptive stats of tr data
        print('  epoch %3d  on train per-pixel VI-loss %.3f  bce %.3f  l1 %.3f' % (
            epoch, tr_loss, tr_bce, tr_l1))

        te_loss, te_l1, te_bce, te_msg = eval_model_on_data(
            tmp_vae_model, 'test', test_loader_in, device, args)
        if epoch == 0:
            print(te_msg) # descriptive stats of test data
        print('  epoch %3d  on test  per-pixel VI-loss %.3f  bce %.3f  l1 %.3f' % (
            epoch, te_loss, te_bce, te_l1))

        ## Write perf metrics to CSV file (so we can easily plot later)
        # Create str repr of architecture size list: [20,30] becomes '[20;30]'
        arch_str = '[' + ';'.join(map(str,args.hidden_layer_sizes)) + ']'
        row_df = pd.DataFrame([[
                epoch, tr_loss, tr_l1, tr_bce, te_loss, te_l1, te_bce,
                arch_str, args.lr, args.q_sigma, args.n_mc_samples]],
            columns=[
                'epoch',
                'tr_vi_loss', 'tr_l1_error', 'tr_bce_error',
                'te_vi_loss', 'te_l1_error', 'te_bce_error',
                'arch_str', 'lr', 'q_sigma', 'n_mc_samples'])
        csv_str = row_df.to_csv(
            None,
            float_format='%.8f',
            index=False,
            header=False if epoch > 0 else True,
            )
        if epoch == 0:
            # At start, write to a clean file with mode 'w'
            with open(f'data/training/{args.filename_prefix}_{args.train_data}_perf_metrics.csv', 'w') as f:
                f.write(csv_str)
        else:
            # Append to existing file with mode 'a'
            with open(f'data/training/{args.filename_prefix}_{args.train_data}_perf_metrics.csv' , 'a') as f:
                f.write(csv_str)

        ## Make pretty plots of random samples in code space decoding into data space
        with torch.no_grad():
            P = int(np.sqrt(model.n_dims_data))
            sample = torch.randn(25, model.n_dims_code).to(device)
            sample = model.decode(sample).cpu()
            save_image(
                sample.view(25, 1, P, P), 
                f'data/samples/{args.filename_prefix}-{args.train_data}-sampled_images-epoch={epoch}.png',
                nrow=5, padding=4)


            model_fpath = f'data/models/{args.filename_prefix}-{args.train_data}-model-epoch={epoch}.pytorch' 
            model.save_to_file(model_fpath)


        print(f"====  done with eval at epoch {epoch}" )

    ## making predictions
    # pred_probas_in = predict_on_data(model, test_loader_in, args)
    # pred_probas_out = predict_on_data(model, test_loader_out, args)
