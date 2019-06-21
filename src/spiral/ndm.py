"""Neural Nonlinear Dynamical Model"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.spiral.dataset import generate_spiral2d
from src.spiral.utils import AverageMeter, log_normal_pdf, normal_kl
from src.spiral.ldm import LDM, merge_inputs, get_parser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NDM(LDM):
    """
    Nonlinear Dynamical Model parameterizes by neural networks.

    Equivalent to LDM but transition and emission transformations
    are governed by nonlinear functions.

    THis is also equivalent to a deep markov model (DMM).
    https://arxiv.org/abs/1609.09869

    y_dim := integer
            number of input dimensions
    x_dim := integer
             number of latent dimensions
    x_emission_dim := integer
                      hidden dimension from y_dim -> x_dim
    x_transition_dim := integer
                        hidden dimension from x_dim -> x_dim
    rnn_dim := integer
               hidden dimension for RNN over y
    rnn_dropout_rate := float [default: 0.]
                        dropout over nodes in RNN
    """
    def __init__(self, y_dim, x_dim, x_emission_dim, x_transition_dim,
                 rnn_dim, rnn_dropout_rate=0.):
        super().__init__(y_dim, x_dim, x_emission_dim, x_transition_dim,
                         rnn_dim, rnn_dropout_rate=rnn_dropout_rate)
        # overwrite with new nonlinear versions:
        # p(x_t|x_t-1)
        self.transistor = Transistor(x_dim, x_transition_dim)
        # p(y_t|x_t)
        self.emitter = Emitter(y_dim, x_dim, x_emission_dim)
        # q(x_t|x_t-1,y_t:T)
        self.combiner = Combiner(x_dim, rnn_dim)


class Emitter(nn.Module):
    """
    Parameterizes `p(y_t | x_t)`.
    
    Args
    ----
    y_dim := integer
             number of input dimensions
    x_dim := integer
             number of latent dimensions
    emission_dim := integer
                    number of hidden dimensions in output 
    """
    def __init__(self, y_dim, x_dim, emission_dim):
        super().__init__()
        self.lin_x_to_hidden = nn.Linear(x_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, y_dim)
    
    def forward(self, x_t):
        h1 = F.relu(self.lin_x_to_hidden(x_t))
        h2 = F.relu(self.lin_hidden_to_hidden(h1))
        return self.lin_hidden_to_input(h2)


class Transistor(nn.Module):
    """
    Parameterizes `p(x_t | x_{t-1})`.

    Args
    ----
    x_dim := integer
             number of latent dimensions
    transition_dim := integer
                      number of hidden dimensions in transistor
    """
    def __init__(self, x_dim, transition_dim):
        super().__init__()
        self.lin_x_to_hidden = nn.Linear(x_dim, transition_dim)
        self.lin_hidden_to_hidden = nn.Linear(transition_dim, transition_dim)
        self.lin_hidden_to_mu = nn.Linear(transition_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(transition_dim, x_dim)
    
    def forward(self, x_t_1):
        h1 = F.relu(self.lin_x_to_hidden(x_t_1))
        h2 = F.relu(self.lin_hidden_to_hidden(h1))
        mu = self.lin_hidden_to_mu(h2)
        logvar = torch.zeros_like(mu)
        # logvar = self.lin_hidden_to_logvar(h1)
        return mu, logvar


class Combiner(nn.Module):
    """
    Parameterizes `q(x_t | x_{t-1}, y_{t:T})`, which is the basic 
    building block of the guide (i.e. the variational distribution). 
    The dependence on `y_{t:T}` is through the hidden state of the RNN 
    (see the PyTorch module `rnn` below)

    Args
    ----
    x_dim := integer
             number of latent dimensions
    rnn_dim := integer
               hidden dimensions of RNN
    """
    def __init__(self, x_dim, rnn_dim):
        super().__init__()
        self.lin_x_to_hidden = nn.Linear(x_dim, rnn_dim)
        self.lin_hidden_to_hidden = nn.Linear(rnn_dim, rnn_dim)
        self.lin_hidden_to_mu = nn.Linear(rnn_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(rnn_dim, x_dim)

    def forward(self, x_t_1, h_rnn):
        # combine the rnn hidden state with a transformed version of z_t_1
        x_input = F.relu(self.lin_x_to_hidden(x_t_1))
        x_input = self.lin_hidden_to_hidden(x_input)
        h_combined = 0.5 * (torch.tanh(x_input) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        x_t_mu = self.lin_hidden_to_mu(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        x_t_logvar = torch.zeros_like(x_t_mu) 
        # x_t_logvar = self.lin_hidden_to_logvar(h_combined)
        # return parameters of normal distribution
        return x_t_mu, x_t_logvar


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=1000, start=0., stop=6 * np.pi, noise_std=.3, a=0., b=.3)
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)

    ndm = NDM(3, 4, 20, 20, 25).to(device)
    optimizer = optim.Adam(ndm.parameters(), lr=args.lr)

    loss_meter = AverageMeter()
    tqdm_pbar = tqdm(total=args.niters)
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        inputs = merge_inputs(samp_trajs, samp_ts)
        outputs = ndm(inputs)
        loss = ndm.compute_loss(inputs, outputs)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        tqdm_pbar.set_postfix({"loss": -loss_meter.avg})
        tqdm_pbar.update()
    tqdm_pbar.close()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    checkpoint_path = os.path.join(args.out_dir, 'checkpoint.pth.tar')
    torch.save({
        'state_dict': ndm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'orig_trajs': orig_trajs,
        'samp_trajs': samp_trajs,
        'orig_ts': orig_ts,
        'samp_ts': samp_ts,
        'model_name': 'ndm',
    }, checkpoint_path)
