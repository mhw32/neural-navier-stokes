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
from rslds import RSLDS

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
    device = torch.device('cuda' if args.cuda else 'cpu')

    print("Using cuda:", args.cuda)
    print("Device:", device)
    print("Params save dir:", args.out_dir)
    print("Plots save dir:", args.plot_dir)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_timesteps = 1000
    train_dataset = BernoulliLorenz(100, num_timesteps, dt=0.01)

    test_num_timesteps = 2000
    test_dataset = BernoulliLorenz(1, test_num_timesteps, dt=0.01)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    #model = RSLDS(2, 1, 10, 100, 10, 10, 1, 20, 20)
    model = RSLDS(2, 1, 3, 100, 10, 3, 1, 20, 20)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    temp, temp_min, temp_anneal_rate = 1.0, 0.1, 0.00003

    step = 0
    best_elbo = sys.maxint
    for i in xrange(5000):
        for batch_idx, data in enumerate(train_loader):
            batch_size = len(data)
            data = data.to(device)

            output = model(data, temp)
            elbo = many_systems_evidence_lower_bound(data, output)

            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            if step % 1 == 0:
                print('step %d: loss = %.4f (temp = %.2f)' % (step, elbo.item(), temp))

            if step % 10 == 0:
                temp = np.maximum(temp * np.exp(-temp_anneal_rate * step), temp_min)

#                for _, test_data in enumerate(test_loader):
#                    test_data = test_data.to(device)
#                    plot_inference(model, test_data, temp, step, args.plot_dir)
#                    plot_generator(model, test_num_timesteps, temp, step, args.plot_dir)

            if elbo.item() < best_elbo:
                best_elbo = elbo.item()
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'epoch_{}_params.pth'.format(i+1)))

            step += 1

