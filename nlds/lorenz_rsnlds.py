from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from datasets import BernoulliLorenz
from elbo import many_systems_evidence_lower_bound
from rsnlds import RSSNLDS

from plot_latent_space import plot_inference, plot_generator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='./')
    parser.add_argument('--plot-dir', type=str, default='./')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda:2' if args.cuda else 'cpu') # hack for now to run on gpu 2

    print("Using cuda:", args.cuda)
    print("Device:", device)
    print("Params save dir:", args.out_dir)
    print("Plots save dir:", args.plot_dir)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    num_timesteps = 1000
    train_dataset = BernoulliLorenz(100, num_timesteps, dt=0.015)

    test_num_timesteps = 2000
    test_dataset = BernoulliLorenz(1, test_num_timesteps, dt=0.01)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    #model = RSSNLDS(2, 1, 10, 100, 10, 10, 20, 20, 64, 64)
    model = RSSNLDS(2, 1, 3, 100, 10, 10, 20, 20, 64, 64)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    temp, temp_min, temp_anneal_rate = 1.0, 0.1, 0.00003

    best_elbo = sys.maxint
    for i in xrange(5000):
        step = 0
        for batch_idx, data in enumerate(train_loader):
            batch_size = len(data)
            data = data.to(device)

            output = model(data, temp)
            elbo = many_systems_evidence_lower_bound(data, output)

            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            print('epoch: %d step %d: loss = %.4f (temp = %.6f)' % (i+1, step+1, elbo.item(), temp))
            temp = np.maximum(temp * np.exp(-temp_anneal_rate * step), temp_min)

#            for _, test_data in enumerate(test_loader):
#                test_data = test_data.to(device)
#                plot_inference(model, test_data, temp, step, args.plot_dir)
#                plot_generator(model, test_num_timesteps, temp, step, args.plot_dir)

            if elbo.item() < best_elbo:
                best_elbo = elbo.item()
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'epoch_{}_step_{}_params.pth'.format(i+1, step+1)))

            step += 1

