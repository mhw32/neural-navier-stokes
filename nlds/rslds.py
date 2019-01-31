from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import gumbel_softmax_sample, gumbel_softmax
from ldm import LDM
from ldm import reverse_sequences_torch


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
    def __init__(self, x_dim, categorical_dim, z_dim, emission_dim):
        super(GaussianEmitter, self).__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim * categorical_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_mu = nn.Linear(emission_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(emission_dim, x_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t):
        h1 = self.relu(self.lin_z_to_hidden(z_t))
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
    x := gaussian latent variable from LDM

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
        self.lin_z_to_hidden = nn.Linear(z_dim * categorical_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim * categorical_dim)
        self.tanh = nn.Tanh()

    def forward(self, z_t_1, x_rnn):
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + x_rnn)
        z_t_logit = self.lin_hidden_to_loc(h_combined)

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
        self.lin_compress_x_z_to_z = nn.Linear(z_dim * categorical_dim + x_dim, z_dim * categorical_dim)
        self.lin_gate_z_to_hidden = nn.Linear(z_dim * categorical_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim * categorical_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim * categorical_dim, transition_dim)
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


class RSLDS(nn.Module):
    def __init__(
        self,
        categorical_dim,
        z_dim,
        x_dim,
        y_dim,
        z_emission_dim,
        x_transition_dim,
        z_transition_dim,
        x_rnn_dim,
        y_rnn_dim,
        x_rnn_dropout_rate=0.0,
        y_rnn_dropout_rate=0.0
    ):
        super(RSLDS, self).__init__()
        self.categorical_dim = categorical_dim
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_emission_dim = z_emission_dim
        self.x_transition_dim = x_transition_dim
        self.z_transition_dim = z_transition_dim
        self.x_rnn_dim = x_rnn_dim
        self.x_rnn_dropout_rate = x_rnn_dropout_rate   
        self.y_rnn_dim = y_rnn_dim
        self.y_rnn_dropout_rate = y_rnn_dropout_rate   

        # RNNs over continuous latent variables, x
        self.x_rnns = nn.ModuleList([
            nn.RNN(
                input_size = self.x_dim,
                hidden_size = self.x_rnn_dim,
                nonlinearity = 'relu',
                batch_first = True,
                bidirectional = False,
                num_layers = 1,
                dropout = self.x_rnn_dropout_rate
            )
            for _ in xrange(self.categorical_dim)
        ])

        # Parameterize p(z_t | z_{t-1}, x_{t-1})
        self.z_trans = CategoricalGatedTransition(
            self.categorical_dim,
            self.z_dim,
            self.x_dim,
            self.z_transition_dim
        )

        # Parameterize q(z_t | z_{t-1}, RNN(x_{t:T}))
        self.z_combiner = CategoricalCombiner(self.categorical_dim, self.z_dim, self.x_rnn_dim)

        # Function to combine the different g(RNN(x_{1:T}_1), ..., RNN(x_{1:T}_K))
        self.z_cat_combiner = nn.Linear(self.x_rnn_dim * self.categorical_dim, self.x_rnn_dim)

        # function to get x_t = p(x_t | z_t)
        self.x_emitter = GaussianEmitter(
            self.x_dim,
            self.categorical_dim,
            self.z_dim,
            self.z_emission_dim
        )

        # Define (trainable) parameters z_0 and z_q_0 that help define
        # the probability distributions p(z_1) and q(z_1)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim * self.categorical_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_dim * self.categorical_dim))

        # Define a (trainable) parameter for the initial hidden state of each RNN
        self.h_0s = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, self.x_rnn_dim)) for _ in xrange(self.categorical_dim)
        ])

        # Define linear dynamical systems for each state
        self.systems = nn.ModuleList([
            LDM(
                self.y_dim,
                self.x_dim,
                self.y_dim,
                self.x_transition_dim,
                self.y_rnn_dim,
                rnn_dropout_rate=self.y_rnn_dropout_rate
            )
            for _ in xrange(self.categorical_dim)
        ])

    def reparameterize(self, logit, temperature):
        b, n = logit.size(0), logit.size(1)
        logit = logit.view(b, n // self.categorical_dim, self.categorical_dim)
        return gumbel_softmax_sample(logit, temperature).squeeze(1)

    def gaussian_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def build_gaussian_mixture(self, weights, samples, means, logvars):
        assert len(samples) == len(means)
        assert len(means) == len(logvars)
        K = len(samples)
        mixture_samples = 0
        mixture_mean = 0
        mixture_var = 0
        for i in xrange(K):
            mixture_samples += (weights[i] * samples[i])
            mixture_mean += (weights[i] * means[i])
            mixture_var += (weights[i]**2 * torch.exp(logvars[i])**2)
        mixture_logvar = torch.log(mixture_var)
        
        return mixture_samples, mixture_mean, mixture_logvar

    def inference_network(self, data, temperature):
        # data := y (batch_size, time_steps, dimension)
        batch_size = data.size(0)
        T = data.size(1)

        # shape: list of length K; each element is (batch_size, T, x_rnn_dim)
        x_rnn_output_K = []

        # shape: list of length K; each element is (batch_size, T, x_dim)
        q_x_K, q_x_mu_K, q_x_logvar_K = [], [], []

        for i in xrange(self.categorical_dim):
            system_i = self.systems[i]
            # q_x: (batch_size, T, x_dim)
            q_x, q_x_mu, q_x_logvar = system_i.inference_network(data)
            q_x_reversed = system_i.reverse_data(q_x)

            q_x_seq_lengths = [T for _ in xrange(batch_size)]
            q_x_seq_lengths = np.array(q_x_seq_lengths)
            q_x_seq_lengths = torch.from_numpy(q_x_seq_lengths).long()
            q_x_seq_lengths = q_x_seq_lengths.to(data.device)

            h_0_contig = self.h_0s[i].expand(1, batch_size, self.x_rnns[i].hidden_size).contiguous()

            x_rnn_i_output, _ = self.x_rnns[i](q_x_reversed, h_0_contig)
            x_rnn_i_output = reverse_sequences_torch(x_rnn_i_output, q_x_seq_lengths)

            x_rnn_output_K.append(x_rnn_i_output)

            q_x_K.append(q_x)
            q_x_mu_K.append(q_x_mu)
            q_x_logvar_K.append(q_x_logvar)

        # reshape into (batch_size, T, x_rnn_dim * categorical_dim)
        x_rnn_output_K = torch.cat(x_rnn_output_K, dim=2)

        # initialize categorical distribution
        z_prev_logits = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        # use gumble softmax to reparameterize
        z_prev = self.reparameterize(z_prev_logits, temperature)

        z_sample_T, z_logit_T = [], []
        x_sample_T, x_mu_T, x_logvar_T = [], [], []
        
        for t in xrange(1, T + 1):
            x_rnn_output = F.relu(self.z_cat_combiner(x_rnn_output_K[:, t - 1, :]))
            z_logit = self.z_combiner(z_prev, x_rnn_output)
            z_t = self.reparameterize(z_logit, temperature)

            q_x_t, q_x_mu_t, q_x_logvar_t = [], [], []

            for i in xrange(batch_size):
                weights_ti = z_t[i]
                samples_ti = [q_x_K[j][i, t-1, :] for j in xrange(self.categorical_dim)]
                means_ti = [q_x_mu_K[j][i, t-1, :] for j in xrange(self.categorical_dim)]
                logvars_ti = [q_x_logvar_K[j][i, t-1, :] for j in xrange(self.categorical_dim)]
                samples_mix_ti, means_mix_ti, logvars_mix_ti = self.build_gaussian_mixture(
                    weights_ti, samples_ti, means_ti, logvars_ti)
                q_x_t.append(samples_mix_ti)
                q_x_mu_t.append(means_mix_ti)
                q_x_logvar_t.append(logvars_mix_ti)

            q_x_t = torch.stack(q_x_t)
            q_x_mu_t = torch.stack(q_x_mu_t)
            q_x_logvar_t = torch.stack(q_x_logvar_t)

            z_sample_T.append(z_t)
            z_logit_T.append(z_logit)

            x_sample_T.append(q_x_t)
            x_mu_T.append(q_x_mu_t)
            x_logvar_T.append(q_x_logvar_t)

            z_prev = z_t  # update for next iter

        x_sample_T = torch.stack(x_sample_T).permute(1, 0, 2)  # (batch_size, T, x_dim)
        x_mu_T = torch.stack(x_mu_T).permute(1, 0, 2)
        x_logvar_T = torch.stack(x_logvar_T).permute(1, 0, 2)

        z_sample_T = torch.stack(z_sample_T).permute(1, 0, 2)  # (batch_size, T, x_dim)
        z_logit_T = torch.stack(z_logit_T).permute(1, 0, 2)
        
        return z_sample_T, z_logit_T, x_sample_T, x_mu_T, x_logvar_T

    # Define a generative model over p(z_{t} | z_{t-1})
    def generative_model(self, batch_size, T, temperature):
        # shape: list of length K; each element is (batch_size, T, x_dim)
        q_x_K, q_x_mu_K, q_x_logvar_K = [], [], []

        for i in xrange(self.categorical_dim):
            system_i = self.systems[i]
            q_x, q_x_mu, q_x_logvar = system_i.generative_model(batch_size, T)

            q_x_K.append(q_x)
            q_x_mu_K.append(q_x_mu)
            q_x_logvar_K.append(q_x_logvar)

        # initialize gumbel distribution
        z_logits = self.z_0.expand(batch_size, self.z_0.size(0))
        # use gumble softmax to reparameterize
        z_prev = self.reparameterize(z_logits, temperature)

        # build 0th x_prev element by element depending on the dynamic system
        t = 0; x_prev = []
        for i in xrange(batch_size):
            weights_ti = z_prev[i]
            samples_ti = [q_x_K[j][i, t, :] for j in xrange(self.categorical_dim)]
            means_ti = [q_x_mu_K[j][i, t, :] for j in xrange(self.categorical_dim)]
            logvars_ti = [q_x_logvar_K[j][i, t, :] for j in xrange(self.categorical_dim)]
            samples_mix_ti, _, _ = self.build_gaussian_mixture(
                weights_ti, samples_ti, means_ti, logvars_ti)
            x_prev_i = samples_mix_ti
            x_prev.append(x_prev_i)

        x_prev = torch.stack(x_prev)

        z_sample_T, z_logit_T = [], []
        x_sample_T, x_mu_T, x_logvar_T = [], [], []

        for t in xrange(1, T + 1):
            z_logit = self.z_trans(z_prev, x_prev)
            z_t = self.reparameterize(z_logit, temperature)

            z_sample_T.append(z_t)
            z_logit_T.append(z_logit)

            q_x_t, q_x_mu_t, q_x_logvar_t = [], [], []

            for i in xrange(batch_size):
                weights_ti = z_t[i]
                samples_ti = [q_x_K[j][i, t - 1, :] for j in xrange(self.categorical_dim)]
                means_ti = [q_x_mu_K[j][i, t - 1, :] for j in xrange(self.categorical_dim)]
                logvars_ti = [q_x_logvar_K[j][i, t - 1, :] for j in xrange(self.categorical_dim)]
                samples_mix_ti, means_mix_ti, logvars_mix_ti = self.build_gaussian_mixture(
                    weights_ti, samples_ti, means_ti, logvars_ti)
                q_x_t.append(samples_mix_ti)
                q_x_mu_t.append(means_mix_ti)
                q_x_logvar_t.append(logvars_mix_ti)

            q_x_t = torch.stack(q_x_t)
            q_x_mu_t = torch.stack(q_x_mu_t)
            q_x_logvar_t = torch.stack(q_x_logvar_t)

            x_sample_T.append(q_x_t)
            x_mu_T.append(q_x_mu_t)
            x_logvar_T.append(q_x_logvar_t)

            z_prev = z_t
            x_prev = q_x_t

        x_sample_T = torch.stack(x_sample_T).permute(1, 0, 2)  # (batch_size, T, x_dim)
        x_mu_T = torch.stack(x_mu_T).permute(1, 0, 2)
        x_logvar_T = torch.stack(x_logvar_T).permute(1, 0, 2)

        z_sample_T = torch.stack(z_sample_T).permute(1, 0, 2)  # (batch_size, T, x_dim)
        z_logit_T = torch.stack(z_logit_T).permute(1, 0, 2)
        
        return z_sample_T, z_logit_T, x_sample_T, x_mu_T, x_logvar_T

    def forward(self, data, temperature):
        batch_size, T, _ = data.size()
        q_z, q_z_logit, q_x, q_x_mu, q_x_logvar = self.inference_network(data, temperature)
        p_z, p_z_logit, p_x, p_x_mu, p_x_logvar = self.generative_model(batch_size, T, temperature)

        y_emission_probs = []
        x_emission_mu, x_emission_logvar = [], []

        for t in xrange(1, T + 1):
            z_t = p_z[:, t - 1]
            y_emission_probs_t = []
            
            for i in xrange(batch_size):
                weights_ti = z_t[i]
                y_emission_probs_t_i = 0
                
                for j in xrange(self.categorical_dim):
                    y_emission_probs_t_i += (weights_ti[j] * self.systems[j].emitter(p_x[i,t-1,:])

                y_emission_probs_t.append(y_emission_probs_t_i)
            y_emission_probs_t = torch.stack(y_emission_probs_t)

            x_emission_mu.append(p_x_mu[:,t-1,:])
            x_emission_logvar.append(p_x_logvar[:,t-1,:])
            y_emission_probs.append(y_emission_probs_t)

        y_emission_probs = torch.stack(y_emission_probs)
        x_emission_mu = torch.stack(x_emission_mu)
        x_emission_logvar = torch.stack(x_emission_logvar)

        y_emission_probs = y_emission_probs.permute(1, 0, 2)
        x_emission_mu = x_emission_mu.permute(1, 0, 2)
        x_emission_logvar = x_emission_logvar.permute(1, 0, 2)

        output = {
            'q_z': q_z,
            'q_z_logit': q_z_logit,
            'q_x': q_x,
            'q_x_mu': q_x_mu,
            'q_x_logvar': q_x_logvar,
            'p_z': p_z,
            'p_z_logit': p_z_logit,
            'p_x': p_x,
            'p_x_mu': p_x_mu,
            'p_x_logvar': p_x_logvar,
            'x_emission_mu': x_emission_mu,
            'x_emission_logvar': x_emission_logvar,
            'y_emission_probs': y_emission_probs,
        }

        return output

