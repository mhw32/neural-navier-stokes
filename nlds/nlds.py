r"""Switching Recurrent Nonlinear Dynamical System."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn

from dmm import DMM
from utils import gumbel_softmax


class GaussianEmitter(nn.Module):
    r"""Parameterizes the gaussian observation likelihood `p(x_t | z_t)`

    @param x_dim: integer
                  number of gaussian latent dimensions
    @param z_dim: integer
                  number of categorical latent dimensions
    @param emission_dim: integer
                         number of hidden dimensions in output 
                         neural network
    """
    def __init__(self, x_dim, z_dim, emission_dim):
        super(GaussianEmitter, self).__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_mu = nn.Linear(emission_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(emission_dim, x_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t):
        h1 = self.relu(self.lin_x_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        x_mu = self.lin_hidden_to_mu(h2)
        x_logvar = self.lin_hidden_to_logvar(h2)
        
        return x_mu, x_logvar


class CategoricalCombiner(nn.Module):
    r"""Parameterizes `q(z_t | z_{t-1}, x_{t:T})`.

    We assume that we pass an RNN through the latent variables 
    of the deep markov model.

    This is highly modeled after the Combiner in ./dmm.py

    z := categorical variable in NLDS
    x := gaussian latent variable from DMM

    @param categorical_dim: integer
                            number of categories
                            full dimension is categorical_dim * z_dim
    @param z_dim: integer
                  number of latent dimensions
    @param rnn_dim: integer
                    hidden dimensions of RNN
    """
    def __init__(self, categorical_dim, z_dim, rnn_dim):
        super(CategoricalCombiner, self).__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim * categorical_dim)
        self.tanh = nn.Tanh()

    def forward(self, z_t_1, x_rnn, temperature):
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + x_rnn)
        z_t_logit = self.lin_hidden_to_loc(h_combined)
        # note: logit means no gumble-softmax-reparameterization yet

        return z_t_logit


class CategoricalGatedTransition(nn.Module):
    r"""Parameterizes the categorical latent transition probability `p(z_t | z_{t-1}, x_{t-1})`
    We will merely add an extra layer going from [z_dim + x_dim --> z_dim]

    @param categorical_dim: integer
                            number of categories
    @param z_dim: integer
                  number of latent dimensions
    @param transition_dim: integer
                           number of transition dimensions
    """
    def __init__(self, categorical_dim, z_dim, x_dim, transition_dim):
        super(CategoricalGatedTransition, self).__init__()
        self.lin_compress_x_z_to_z = nn.Linear(z_dim + x_dim, z_dim)
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim * categorical_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim * categorical_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim * categorical_dim)
        
        self.lin_z_to_loc.weight.data = torch.eye(z_dim * categorical_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim * categorical_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t_1, x_t_1):
        z_x_t_1 = torch.cat([z_t_1, x_t_1], dim=1)
        z_t_1 = self.relu(self.lin_compress_x_z_to_z(z_x_t_1))

        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        
        z_t_logit = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # note: logit means no gumble-softmax-reparameterization yet

        return z_t_logit


class RSSNLDS(nn.Module):
    r"""Variational Recurrent Switching State NonLinear Dynamical System.
    
    "Categorical" distribution over several nonlinear dynamical systems,
    each of which is modeled by a DMM. 

    Natural generalization to https://arxiv.org/pdf/1610.08466.pdf.

    We want this to be trainable end-to-end by gradient descent. So, we 
    will rely on a Gumbel softmax parameterization of a categorical.

    z := categorical variable in NLDS
    x := gaussian latent variable from DMM
    y := bernoulli observed variable from DMM

    Our notation here uses <variable>2 to represent discrete states
    and <variable> to represent continuous states.
    """
    def __init__(self, temperature, categorical_dim, z_dim, x_dim, y_dim, 
                 z_emission_dim, x_emission_dim, z_transition_dim, x_transition_dim, 
                 z_rnn_dim, x_rnn_dim, z_rnn_dropout_rate=0.0, x_rnn_dropout_rate=0.0):
        super(RSSNLDS, self).__init__()
        self.temperature = temperature
        self.categorical_dim = categorical_dim
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_emission_dim = z_emission_dim
        self.x_emission_dim = x_emission_dim
        self.z_transition_dim = z_transition_dim
        self.x_transition_dim = x_transition_dim
        self.z_rnn_dim = z_rnn_dim
        self.x_rnn_dim = x_rnn_dim
        self.z_rnn_dropout_rate = z_rnn_dropout_rate   
        self.x_rnn_dropout_rate = x_rnn_dropout_rate   

        # this is an RNN over the Gaussian latent dimensions in DMM
        self.x_rnn = nn.RNN(input_size=self.x_dim, hidden_size=self.z_rnn_dim, nonlinearity='relu',
                            batch_first=True, bidirectional=False, num_layers=1,
                            dropout=self.z_rnn_dropout_rate)

        # function to get z_t = p(z_t-1)
        self.z_trans = CategoricalGatedTransition(
            self.categorical_dim, self.z_dim, self.x_dim, self.z_transition_dim)

        # function to get z_t = q(z_t-1, x_t-1)
        self.z_combiner = CategoricalCombiner(
            self.categorical_dim, self.z_dim, self.rnn_dim)

        # function to get x_t = p(x_t | z_t)
        self.x_emitter = GaussianEmitter(self.x_dim, self.z_dim, self.z_emission_dim)

        # define (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim * self.categorical_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_dim * self.categorical_dim))
        
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.z_rnn_dim))

        # define |categorical| nonlinear dynamic systems
        self.systems = nn.ModuleList([   
            DMM(self.y_dim, self.x_dim, self.x_emission_dim, self.x_transition_dim, 
                self.x_rnn_dim, rnn_dropout_rate=self.x_rnn_dropout_rate)
            for _ in xrange(self.categorical_dim)
        ])

    def reparameterize(self, logit, temperature):
        return gumbel_softmax(logit, temperature)

    def gaussian_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # define an inference network q(z_{1:T}|f(x_{1:T})^(1:N)) where T is maximum length
    # where f(.) is a recurrent neural network. NOTE: this is the same inference 
    # network used in the DMM. The major difference is that each sampled z_t unlocks
    # a different sequence x_{1:T}^(j).
    def inference_network(self, data, temperature):
        # data := y
        batch_size = data.size(0)
        T_max = data.size(1)

        # save parameters of q_z here
        z_list = []
        z_logit_list = []

        # save parameters of q_x here
        x_list = []
        x_mu_list = []
        x_logvar_list = []

        # tmp storage
        x_list_tmp = []
        x_mu_list_tmp = []
        x_logvar_list_tmp = []

        # HACK: quite expensive, but lets compute for all systems otherwise we 
        # have to deal with nasty batching and I dont want for MVP
        for i in xrange(self.categorical_dim):
            system_i = self.systems[i]
            # q_x_list_i is a list of samples of variable x
            q_x_list_i, q_x_mu_list_i, q_x_logvar_list_i = system_i.inference_network(data)

            # for now, store entire lists of q's. We will choose the right ones later
            x_list_tmp.append(q_x_list_i)
            x_mu_list_tmp.append(q_x_mu_list_i)
            x_logvar_list_tmp.append(q_x_logvar_list_i)

        # initialize categorical distribution
        z_prev_logits = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        # use gumble softmax to reparameterize
        z_prev = self.reparameterize(z_prev_logits, temperature)

        t = 0  # grab x at timestep 1 from system <z_prev> for each item in batch
        x_prev = []
        for i in xrange(batch_size):
            cat = z_prev[i].item()
            # x_prev_i is a Gaussian latent variable
            x_prev_i = x_list_tmp[cat][t][i].unsqueeze(0)
            x_prev.append(x_prev_i)
        x_prev = torch.cat(x_prev, dim=0)

        # initialize RNN
        x_rnn_hidden = self.h_0.expand(1, batch_size, self.z_rnn.hidden_size)
        x_rnn_hidden = x_rnn_hidden.contiguous()
        x_rnn_output, x_rnn_hidden = self.x_rnn(x_prev, x_rnn_hidden)
        
        for t in xrange(1, T_max + 1):
            z_logit = self.z_combiner(z_prev, x_rnn_output)
            z_t = self.reparameterize(z_logit, temperature)

            z_list.append(z_t)
            z_logit_list.append(z_logit)

            z_prev = z_t  # update for next iter

            # note that is overwrites existing data-structures
            x_prev = []
            x_mu_prev = []
            x_logvar_prev = []

            for i in xrange(batch_size):
                cat = z_prev[i].item()
                # x_prev_i is a Gaussian latent variable
                x_prev_i = x_list_tmp[cat][t][i].unsqueeze(0)
                x_mu_i = x_mu_list_tmp[cat][t][i].unsqueeze(0)
                x_logvar_i = x_logvar_list_tmp[cat][t][i].unsqueeze(0)
                
                x_prev.append(x_prev_i)
                x_mu_prev.append(x_mu_i)
                x_logvar_prev.append(x_logvar_i)

            x_prev = torch.cat(x_prev, dim=0)
            x_mu_prev = torch.cat(x_mu_prev, dim=0)
            x_logvar_prev = torch.cat(x_logvar_prev, dim=0)

            x_list.append(x_prev)
            x_mu_list.append(x_mu_prev)
            x_logvar_list.append(x_logvar_prev)

            # update for next iter
            x_rnn_output, x_rnn_hidden = self.x_rnn(x_prev, x_rnn_hidden)
        
        return z_list, z_logit_list, x_list, x_mu_list, x_logvar_list

    # define a generative model over p(z_{t}|z_{t-1})
    def generative_model(self, batch_size, T_max, temperature):
        # save parameters for q_z
        z_list = []
        z_logit_list = []

        # save parameters for q_x
        x_list = []
        x_mu_list = []
        x_logvar_list = []

        # tmp storage
        x_list_tmp = []
        x_mu_list_tmp = []
        x_logvar_list_tmp = []

        for i in xrange(self.categorical_dim):
            system_i = self.systems[i]
            q_x_list_i, q_x_mu_list_i, q_x_logvar_list_i = \
                system_i.generative_model(batch_size, T_max)

            # for now, store entire lists of q's. We will choose the right ones later
            x_list_tmp.append(q_x_list_i)
            x_mu_list_tmp.append(q_x_mu_list_i)
            x_logvar_list_tmp.append(q_x_logvar_list_i)

        z_prev = self.z_0.expand(batch_size, self.z_0.size(0))

        t = 0
        x_prev = []
        for i in xrange(batch_size):
            cat = z_prev[i].item()
            # x_prev_i is a Gaussian latent variable
            x_prev_i = x_list_tmp[cat][t][i].unsqueeze(0)
            x_prev.append(x_prev_i)
        x_prev = torch.cat(x_prev, dim=0)

        for t in xrange(1, T_max + 1):
            z_logit = self.z_trans(z_prev, x_prev)
            z_t = self.reparameterize(z_logit, temperature)

            z_list.append(z_t)
            z_logit_list.append(z_logit)

            z_prev = z_t

            # note that is overwrites existing data-structures
            x_prev = []
            x_mu_prev = []
            x_logvar_prev = []

            for i in xrange(batch_size):
                cat = z_prev[i].item()
                # x_prev_i is a Gaussian latent variable
                x_prev_i = x_list_tmp[cat][t][i].unsqueeze(0)
                x_mu_i = x_mu_list_tmp[cat][t][i].unsqueeze(0)
                x_logvar_i = x_logvar_list_tmp[cat][t][i].unsqueeze(0)
                
                x_prev.append(x_prev_i)
                x_mu_prev.append(x_mu_i)
                x_logvar_prev.append(x_logvar_i)

            x_prev = torch.cat(x_prev, dim=0)
            x_mu_prev = torch.cat(x_mu_prev, dim=0)
            x_logvar_prev = torch.cat(x_logvar_prev, dim=0)

            x_list.append(x_prev)
            x_mu_list.append(x_mu_prev)
            x_logvar_list.append(x_logvar_prev)

        return z_list, z_logit_list, x_list, x_mu_list, x_logvar_list
        
    def forward(self, data, temperature):
        batch_size, T_max, _ = data.size()
        q_z_list, q_z_logit_list, q_x_list, q_x_mu_list, q_x_logvar_list = \
            self.inference_network(data, temperature)
        p_z_list, p_z_logit_list, p_x_list, p_x_mu_list, p_x_logvar_list = \
            self.generative_model(batch_size, T_max, temperature)
        
        x_emission_mu_list, x_emission_logvar_list = [], []
        y_emission_probs_list = []
        
        for t in xrange(1, T_max + 1):
            z_t = q_z_list[t]
            x_emission_mu_t, x_emission_logvar_t = self.x_emitter(z_t)
            x_emission_t = self.gaussian_reparameterize(
                x_emission_mu_t, x_emission_logvar_t)

            y_emission_probs_t = []
            for i in xrange(batch_size):
                system_i = self.systems[z_t[i].item()]
                y_emission_probs_t_i = system_i.emitter(x_emission_t)[i].unsqueeze(1)
                y_emission_probs_t.append(y_emission_probs_t_i)
            y_emission_probs_t = torch.cat(y_emission_probs_t, dim=0)

            x_emission_mu_list.append(x_emission_mu_t)
            x_emission_logvar_list.append(x_emission_logvar_t)
            y_emission_probs_list.append(y_emission_probs_t)

        output = {
            'q_z': q_z_list,
            'q_z_logit': q_z_logit_list,
            'q_x': q_x_list,
            'q_x_mu': q_x_mu_list,
            'q_x_logvar': q_x_logvar_list,
            'p_z': p_z_list,
            'p_z_logit': p_z_logit_list,
            'p_x': p_x_list,
            'p_x_mu': p_x_mu_list,
            'p_x_logvar': p_x_logvar_list,
            'x_emission_mu': x_emission_mu_list,
            'x_emission_logvar': x_emission_logvar_list,
            'y_emission_probs': y_emission_probs_list,
            'T_max': T_max,
        }
        
        return output
