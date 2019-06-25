"""Neural Switching-state Nonlinear Dynamical Model"""

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
from src.spiral.ldm import reverse_sequences_torch, merge_inputs
from src.spiral.ndm import NDM
from src.spiral.sldm import SLDM
from src.spiral.utils import (AverageMeter, log_normal_pdf, normal_kl, gumbel_softmax,
                              log_mixture_of_normals_pdf, log_gumbel_softmax_pdf)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SNDM(SLDM):
    """
    Switching-State Nonlinear Dynamical Model parameterizes by neural networks.

    We will only approximately be switching state by doing Gumbel-Softmax 
    with a 1 dimensional categorical variable with n_states.

    Equivalent to SNDM but transition and emission transformations
    are governed by nonlinear functions.

    We assume p(z_t | z_{t-1}), p(x_t | x_{t-1}), and p(y_t | x_t) are affine.

    n_states := integer
                number of states
    y_dim := integer
            number of input dimensions
    x_dim := integer
             number of latent dimensions
    x_emission_dim := integer
                      hidden dimension from y_dim -> x_dim
    z_emission_dim := integer
                      hidden dimension from x_dim -> z_dim
    x_transition_dim := integer
                        hidden dimension from x_dim -> x_dim
    z_transition_dim := integer
                        hidden dimension from z_dim -> z_dim
    y_rnn_dim := integer
                 hidden dimension for RNN over y
    x_rnn_dim := integer
                 hidden dimension for RNN over x
    y_rnn_dropout_rate := float [default: 0.]
                          dropout over nodes in RNN
    x_rnn_dropout_rate := float [default: 0.]
                          dropout over nodes in RNN
    """
    def __init__(self, n_states, y_dim, x_dim, x_emission_dim, z_emission_dim, 
                 x_transition_dim, z_transition_dim, y_rnn_dim, x_rnn_dim, 
                 y_rnn_dropout_rate=0., x_rnn_dropout_rate=0.):
        super().__init__(n_states, y_dim, x_dim, x_emission_dim, z_emission_dim,
                         x_transition_dim, z_transition_dim, y_rnn_dim, x_rnn_dim,
                         y_rnn_dropout_rate=y_rnn_dropout_rate,
                         x_rnn_dropout_rate=x_rnn_dropout_rate)
        self.n_states = n_states  # can also call this z_dim
        self.y_dim, self.x_dim = y_dim, x_dim

        # Define (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        self.z_0 = nn.Parameter(torch.randn(n_states))
        self.z_q_0 = nn.Parameter(torch.randn(n_states))

        # Define a (trainable) parameter for the initial hidden state of each RNN
        self.h_0s = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, x_rnn_dim))
                                      for _ in range(n_states)])

        # RNNs over continuous latent variables, x
        self.x_rnns = nn.ModuleList([
            nn.RNN(x_dim, x_rnn_dim, nonlinearity='relu', 
                   batch_first=True, dropout=x_rnn_dropout_rate)
            for _ in range(n_states)
        ])

        # p(z_t|z_t-1)
        self.state_transistor = StateTransistor(n_states, z_transition_dim)
        # p(x_t|z_t)
        self.state_emitter = StateEmitter(x_dim, n_states, z_emission_dim)
        # q(z_t|z_t-1,x_t:T)
        self.state_combiner = StateCombiner(n_states, x_rnn_dim)
        
        self.state_downsampler = StateDownsampler(x_rnn_dim, n_states)

        # initialize a bunch of systems
        self.systems = nn.ModuleList([
            NDM(y_dim, x_dim, x_emission_dim, x_transition_dim,
                y_rnn_dim, rnn_dropout_rate=y_rnn_dropout_rate)
            for _ in range(n_states)
        ])


class StateEmitter(nn.Module):
    """
    Parameterizes `p(x_t | z_t)`.

    Args
    ----
    x_dim := integer
             number of dimensions over latents
    z_dim := integer
             number of dimensions over states
    emission_dim := integer
                    number of hidden dimensions to use in generating x_t 
    """
    def __init__(self, x_dim, z_dim, emission_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_mu = nn.Linear(emission_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(emission_dim, x_dim)
    
    def forward(self, z_t):
        h1 = F.relu(self.lin_z_to_hidden(z_t))
        h2 = F.relu(self.lin_hidden_to_hidden(h1))
        mu = self.lin_hidden_to_mu(h2)
        logvar = torch.zeros_like(mu)
        # logvar = self.lin_hidden_to_logvar(h1)
        return mu, logvar


class StateTransistor(nn.Module):
    """
    Parameterizes `p(z_t | z_{t-1})`. 

    Args
    ----
    z_dim := integer
              number of state dimensions
    transition_dim := integer
                      number of hidden dimensions in transistor
    """
    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_hidden_to_hidden = nn.Linear(transition_dim, transition_dim)
        self.lin_hidden_to_logits = nn.Linear(transition_dim, z_dim)

    def forward(self, z_t_1):
        h1 = F.relu(self.lin_z_to_hidden(z_t_1))
        h2 = F.relu(self.lin_hidden_to_hidden(h1))
        logits = self.lin_hidden_to_logits(h2)
        return logits


class StateCombiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x^1_{t:T}, ..., x^K_{t:T})`
        Since we don't know which system is the most relevant, we
        need to give the inference network all the possible info.

    Args
    ----
    z_dim := integer
             number of latent dimensions
    rnn_dim := integer
               hidden dimensions of RNN
    """
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_hidden = nn.Linear(rnn_dim, rnn_dim)
        self.lin_hidden_to_logits = nn.Linear(rnn_dim, z_dim)

    def forward(self, z_t_1, h_rnn):
        # combine the rnn hidden state with a transformed version of z_t_1
        z_input = F.relu(self.lin_z_to_hidden(z_t_1))
        z_input = self.lin_hidden_to_hidden(z_input)
        h_combined = 0.5 * (torch.tanh(z_input) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        z_t_logits = self.lin_hidden_to_logits(h_combined)
        return z_t_logits


class StateDownsampler(nn.Module):
    """Downsample f(x^1_{t:T}, ..., x^K_{t:T}) to a reasonable size."""
    def __init__(self, rnn_dim, n_states):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rnn_dim * n_states, rnn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_dim, rnn_dim))
    
    def forward(self, x):
        return self.net(x)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='./')
    return parser


def visualize(sndm, orig_trajs, orig_ts, samp_trajs):
    device = orig_trajs.device
    orig_ts = torch.from_numpy(orig_ts).float().to(device)
    T = orig_ts.size(0)

    with torch.no_grad():
        # MODE 1
        # ------
        # get reconstructions with teacher forcing
        # inputs = merge_inputs(orig_trajs, orig_ts)
        inputs = orig_trajs
        outputs = sndm(inputs, 0.01)  # almost 0.
        recon_trajs = outputs['y_emission_mu_1_to_K'][:, :, :, :2]  # ignore time dim

        # TODO: extrapolations by sequential generation

    # plot first 100 examples
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(10):
        for j in range(10):
            index = 10*i + j
            # true trajectory
            axes[i][j].plot(orig_trajs[index][:, 0].cpu().numpy(), 
                            orig_trajs[index][:, 1].cpu().numpy(), '-',
                            label='true trajectory')
            # learned trajectory (teacher-forcing)
            axes[i][j].plot(recon_trajs[index][:, 0].cpu().numpy(), 
                            recon_trajs[index][:, 1].cpu().numpy(), '-',
                            label='teacher forcing')
            axes[i][j].plot(samp_trajs[index][:, 0].cpu().numpy(), 
                            samp_trajs[index][:, 1].cpu().numpy(), 
                            'o', markersize=1, label='dataset')
    axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-4, -0.12), 
                              ncol=5, fontsize=20)
    plt.savefig('./vis_sndm_recons.pdf', dpi=500)

    with torch.no_grad():
        # MODE 2
        # ------
        # sample x0, use transition to get to xt, use generator to yt
        z_prev = sndm.z_0.expand(100, sndm.z_0.size(0))
        z_samples, x_samples, y_samples = [], [], []
        for t in range(T):
            z_t_logits = sndm.state_transistor(z_prev)
            z_t = sndm.gumbel_softmax_reparameterize(z_t_logits, 0.5)
            x_t_mu, x_t_logvar = sndm.state_emitter(z_t)
            x_t = sndm.gaussian_reparameterize(x_t_mu, x_t_logvar)
            z_t_hard = torch.argmax(z_t, dim=1)
            
            y_t_mu = []
            for i in range(100):
                y_t_mu_i = sndm.systems[z_t_hard[i].item()].emitter(x_t[i].unsqueeze(0))
                y_t_mu.append(y_t_mu_i)
            y_t_mu = torch.cat(y_t_mu, dim=0)
            y_t_std_ = torch.zeros_like(y_t_mu) + .3
            y_t_logvar = 2. * torch.log(y_t_std_)
            y_t = sndm.reparameterize(y_t_mu, y_t_logvar)
            z_samples.append(z_t_hard.unsqueeze(1).cpu())
            x_samples.append(x_t.unsqueeze(1).cpu())
            y_samples.append(y_t.unsqueeze(1).cpu())
        z_samples = torch.cat(z_samples, dim=1)
        x_samples = torch.cat(x_samples, dim=1)
        y_samples = torch.cat(y_samples, dim=1)
        z_samples = z_samples.numpy()
        x_samples = x_samples.numpy()
        y_samples = y_samples.numpy()

    # plot first 100 examples
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(10):
        for j in range(10):
            index = 10*i + j
            # true trajectory
            axes[i][j].plot(y_samples[index][:, 0], 
                            y_samples[index][:, 1], '-',
                            label='generated observations')
    axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-4, -0.12), 
                              ncol=5, fontsize=20)
    plt.savefig('./vis_sndm_y_samples.pdf', dpi=500)

    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(10):
        for j in range(10):
            index = 10*i + j
            # true trajectory
            axes[i][j].plot(x_samples[index][:, 0], 
                            x_samples[index][:, 1], '-',
                            label='generated latents')
    axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-4, -0.12), 
                              ncol=5, fontsize=20)
    plt.savefig('./vis_sndm_x_samples.pdf', dpi=500)

    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(10):
        for j in range(10):
            index = 10*i + j
            # true trajectory
            axes[i][j].plot(z_samples[index][:, 0], 
                            z_samples[index][:, 1], '-',
                            label='generated categories')
    axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-4, -0.12), 
                              ncol=5, fontsize=20)
    plt.savefig('./vis_sndm_z_samples.pdf', dpi=500)


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

    sndm = SNDM(4, 2, 4, 20, 20, 20, 20, 25, 25).to(device)
    optimizer = optim.Adam(sndm.parameters(), lr=args.lr)
    
    init_temp, min_temp, anneal_rate = 1.0, 0.5, 0.00003

    loss_meter = AverageMeter()
    tqdm_pbar = tqdm(total=args.niters)
    temp = init_temp
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        # inputs = merge_inputs(samp_trajs, samp_ts)
        inputs = samp_trajs
        outputs = sndm(inputs, temp)
        loss = sndm.compute_loss(inputs, outputs, temp)
        loss.backward()
        optimizer.step()
        if itr % 10 == 1:
            temp = np.maximum(temp * np.exp(-anneal_rate * itr), min_temp)
        loss_meter.update(loss.item())
        tqdm_pbar.set_postfix({"loss": -loss_meter.avg, "temp": temp})
        tqdm_pbar.update()
    tqdm_pbar.close()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    checkpoint_path = os.path.join(args.out_dir, 'checkpoint.pth.tar')
    torch.save({
        'state_dict': sndm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'orig_trajs': orig_trajs,
        'samp_trajs': samp_trajs,
        'orig_ts': orig_ts,
        'samp_ts': samp_ts,
        'temp': temp,
        'model_name': 'sndm',
    }, checkpoint_path)
