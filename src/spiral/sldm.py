"""Neural Switching-state Linear Dynamical Model"""

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
from src.spiral.ldm import reverse_sequences_torch
from src.spiral.ldm import LDM, Combiner, Transistor, Emitter
from src.spiral.utils import (AverageMeter, log_normal_pdf, normal_kl,
                              Sparsemax, MixtureOfDiagNormals)


class SLDM(nn.Module):
    """
    Switching-State Linear Dynamical Model parameterizes by neural networks.

    We will only approximately be switching state: use SparseMax to sparsify
    the categorical distribution over states. Reparameterize implicitly since
    the underlying distribution is Gaussian.

    We assume p(z_t | z_{t-1}), p(x_t | x_{t-1}), and p(y_t | x_t) are affine.

    n_states := integer
                number of states
    y_dim := integer
            number of input dimensions
    x_dim := integer
             number of latent dimensions
    z_dim := integer
             number of space dimensions
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
    def __init__(self, n_states, y_dim, x_dim, z_dim, x_emission_dim, z_emission_dim, 
                 x_transition_dim, z_transition_dim, y_rnn_dim, x_rnn_dim, 
                 y_rnn_dropout_rate=0., x_rnn_dropout_rate=0.):
        super().__init__()

        # Define (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        self.z_0 = nn.Parameter(torch.zeros(z_dim * n_states))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim * n_states))

        # Define a (trainable) parameter for the initial hidden state of each RNN
        self.h_0s = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, self.x_rnn_dim))
                                      for _ in range(n_states)])

        # RNNs over continuous latent variables, x
        self.x_rnns = nn.ModuleList([
            nn.RNN(self.x_dim, self.x_rnn_dim, nonlinearity='relu', 
                   batch_first=True, dropout=self.x_rnn_dropout_rate)
            for _ in range(n_states)
        ])

        # p(z_t|z_t-1)
        self.state_transistor = StateTransistor(z_dim, z_transition_dim)
        # p(x_t|z_t)
        self.state_emitter = StateEmitter(x_dim, z_dim, z_emission_dim)
        # q(z_t|z_t-1,x_t:T)
        self.state_combiner = StateCombiner(z_dim, x_rnn_dim)

        self.state_downsampler = StateDownsampler(x_rnn_dim, n_states)

        # initialize a bunch of systems
        self.ldms = nn.ModuleList([
            LDM(y_dim, x_dim, x_emission_dim, x_transition_dim,
                y_rnn_dim, rnn_dropout_rate=y_rnn_dropout_rate)
            for _ in range(n_states)
        ])

        self.sparsemax = Sparsemax()

    def mixture_reparameterize(self, state_logits, component_mu, component_logvar):
        """Reparameterization through a Mixture of Normals.

        Args
        ----
        state_logits     := batch_size x n_states
                            logit over states
        component_mu     := batch_size x n_states x z_dim
                            mu for each Gaussian distribution
        component_logvar := batch_size x n_states x z_dim
                            logvar for each Gaussian distribution

        Return
        ------
        z := batch_size x z_dim
             sample from mixture of normals
        """
        component_sigma = torch.exp(0.5 * component_logvar)
        z = MixtureOfDiagNormals(
            locs=component_mu.view(-1, self.n_states, self.z_dim),
            coord_scale=component_sigma.view(-1, self.n_states, self.z_dim),
            component_logits=state_logits,
        ).rsample()
        return z

    def gaussian_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def inference_network(self, data):
        """Inference network for defining q(z_t|z_t-1,x^1_1:T,...,x^K_1:T)
                                      and q(x_t|x_t-1,y_1:T)
        Procedure:
            - Loop through each state and use the LDM for that state to define
              D^k = q(x^k_1:T|y_1:T)
              We will draw samples from D^k for k=1 to K and use a RNN for each 
              k to summary the samples for timesteps 1 through T
            - Initialize z(0), the first state. 
            - Define q(z_t|z_t-1,x^1_1:T,...,x^K_1:T) using the RNN summaries 
              for each system k
        """
        batch_size = data.size(0)
        T = data.size(1)
        device = data.device

        # go though each system and get q(x_1:T|y_1:T)
        q_x_mu_1_to_K, q_x_logvar_1_to_K = [], []
        x_summary_1_to_K = []

        for i in range(self.n_states):
            ldm_i = self.ldms[i]
            q_x, q_x_mu, q_x_logvar = ldm_i.inference_network(data)
            q_x_reversed = ldm_i.reverse_data(q_x)
            q_x_mu_1_to_K.append(q_x_mu.unsqueeze(1))
            q_x_logvar_1_to_K.append(q_x_logvar.unsqueeze(1))

            # compute and save RNN(x_1,...,x_T) for each system K
            q_x_seq_lengths = np.array([T for _ in range(batch_size)])
            q_x_seq_lengths = torch.from_numpy(q_x_seq_lengths).long().to(device)

            h_0 = self.h_0s[i].expand(1, batch_size, self.x_rnns[i].hidden_size).contiguous()
            x_rnn_i_output, _ = self.x_rnns[i](q_x_reversed, h_0)
            # batch_size x T x x_rnn_dim
            x_rnn_i_output = reverse_sequences_torch(x_rnn_i_output, q_x_seq_lengths)
            x_summary_1_to_K.append(x_rnn_i_output)

        # q_x_mu_1_to_K     := batch_size x K x D
        # q_x_logvar_1_to_K := batch_size x K x D
        q_x_mu_1_to_K = torch.cat(q_x_mu_1_to_K, dim=1)
        q_x_logvar_1_to_K = torch.cat(q_x_logvar_1_to_K, dim=1)
        x_summary_1_to_K = torch.cat(x_summary_1_to_K, dim=2)  # batch_size x T x x_rnn_dim*n_states
        
        # start z_prev from learned initialization z_q_0
        z_prev_logits = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        x_sample, z_sample, z_logits_1_to_T = [], [], []
        for t in range(1, T + 1):
            x_summary = self.state_downsampler(x_summary_1_to_K[:, t - 1, :])
            # infer q(z_t|z_t-1,x^1_1:T,...,x^T_1:T)
            #   we cant actually pass the sampled z_prev, only the logits...
            z_t_logits = self.state_combiner(z_prev_logits, x_summary)
            # sample from the mixture distribution
            x_t = self.mixture_reparameterize(
                self.sparsemax(z_t_logits), q_x_mu_1_to_K, q_x_logvar_1_to_K)
            x_sample.append(x_t)
            z_sample.append(z_t)
            z_logits_1_to_T.append(z_t_logits)
            z_prev = z_t  # update for next iter

        x_sample_1_to_T = torch.stack(x_sample).permute(1, 0, 2)  # batch_size x T x x_dim
        z_sample_1_to_T = torch.stack(z_sample).permute(1, 0, 2)  # batch_size x T x z_dim
        z_logits_1_to_T = torch.stack(z_logits_1_to_T).permute(1, 0, 2)

        return x_sample_1_to_T, q_x_mu_1_to_K, q_x_logvar_1_to_K, z_sample_1_to_T, z_logits_1_to_T

    def prior_network(self, data):
        pass

    def forward(self, data):
        pass

    def compute_loss(self, data, output):
        pass


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
        self.lin_hidden_to_mu = nn.Linear(emission_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(emission_dim, x_dim)
    
    def forward(self, z_t):
        h1 = self.lin_z_to_hidden(z_t)
        mu = self.lin_hidden_to_mu(h1)
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
        self.lin_x_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_hidden_to_mu = nn.Linear(transition_dim, z_dim)
        self.lin_hidden_to_logvar = nn.Linear(transition_dim, z_dim)

    def forward(self, z_t_1):
        h1 = self.lin_x_to_hidden(z_t_1)
        mu = self.lin_hidden_to_mu(h1)
        logvar = torch.zeros_like(mu)
        # logvar = self.lin_hidden_to_logvar(h1)
        return mu, logvar


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
        self.lin_x_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_mu = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_logvar = nn.Linear(rnn_dim, z_dim)

    def forward(self, z_t_1, h_rnn):
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = self.lin_x_to_hidden(z_t_1) + h_rnn
        # use the combined hidden state to compute the mean used to sample z_t
        x_t_mu = self.lin_hidden_to_mu(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        x_t_logvar = torch.zeros_like(x_t_mu) 
        # x_t_logvar = self.lin_hidden_to_logvar(h_combined)
        return x_t_mu, x_t_logvar


class StateDownsampler(nn.Module):
    """Downsample f(x^1_{t:T}, ..., x^K_{t:T}) to a reasonable size."""
    def __init__(self, rnn_dim, n_states):
        super().__init__()
        self.lin = nn.Linear(rnn_dim * n_states, rnn_dim)
    
    def forward(self, x):
        return self.lin(x)
