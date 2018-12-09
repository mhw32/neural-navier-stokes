r"""Implementation is heavily borrowed from Pyro's implementation.

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BernoulliEmitter(nn.Module):
    r"""Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    
    @param input_dim: integer
                      number of input dimensions
    @param z_dim: integer
                  number of latent dimensions
    @param emission_dim: integer
                         number of output dimensions
    """
    def __init__(self, input_dim, z_dim, emission_dim):
        super(BernoulliEmitter, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        r"""Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps


class GatedTransition(nn.Module):
    r"""Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.

    @param z_dim: integer
                  number of latent dimensions
    @param transition_dim: integer
                           number of transition dimensions
    """
    def __init__(self, z_dim, transition_dim):
        super(GatedTransition, self).__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t_1):
        r"""Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        z_t_mu = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        z_t_logvar = self.lin_sig(self.relu(proposed_mean))
        # return parameters of normal distribution
        return z_t_mu, z_t_logvar


class Combiner(nn.Module):
    r"""Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)

    @param z_dim: integer
                  number of latent dimensions
    @param rnn_dim: integer
                    hidden dimensions of RNN
    """
    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()

    def forward(self, z_t_1, h_rnn):
        r"""Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        z_t_mu = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        z_t_logvar = self.lin_hidden_to_scale(h_combined)
        # return parameters of normal distribution
        return z_t_mu, z_t_logvar


class DMM(nn.Module):
    r"""Deep Markov Model.
    
    NOTE: I assume all inputs are the SAME length. We need to fix this later
          but I bet this assumption will make initial development much easier.
    
    @param input_dim: integer
                      number of input dimensions 
    @param z_dim: integer
                  number of latent dimensions
    @param emissions_dim: integer
                          number of output dimensions
    @param transition_dim: integer
                           number of transition dimensions
    @param rnn_dim: integer
                    hidden dimensions of RNN
    @param rnn_dropout_rate: float [default: 0.0]
                             dropout rate for RNN
    """
    def __init__(self, input_dim, z_dim, emission_dim, transition_dim, 
                    rnn_dim, rnn_dropout_rate=0.0):
        super(DMM, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.rnn_dropout_rate = rnn_dropout_rate

        self.emitter = BernoulliEmitter(self.input_dim, self.z_dim, self.emission_dim)
        self.trans = GatedTransition(self.z_dim, self.transition_dim)
        self.combiner = Combiner(self.z_dim, self.rnn_dim)
        self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=self.rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=1,
                          dropout=self.rnn_dropout_rate)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def reverse_data(self, data):
        data_npy = data.cpu().numpy()
        data_reversed_npy = data_npy[:, ::-1, :]
        data_reversed = torch.from_numpy(data_reversed_npy)
        data_reversed = data_reversed.to(data.device)
        return data_reversed

    # define an inference network q(z_{1:T}|x_{1:T}) where T is maximum length
    # FIX: not all inputs are used
    def inference_network(self, data):
        # data: (batch_size, time_steps, dimension)
        # this is the number of time steps we need to process in the mini-batch
        batch_size = data.size(0)
        T_max = data.size(1)
        
        # flip data as RNN takes data in opposing order
        data_reversed = self.reverse_data(data)

        # compute sequence lengths
        data_seq_lengths = [T_max for _ in xrange(batch_size)]
        data_seq_lengths = torch.from_numpy(data_seq_lengths).long()
        data_seq_lengths = data_seq_lengths.to(data.device)

        h_0_contig = self.h_0.expand(
            1, data.size(0), self.rnn.hidden_size).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(data_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = pad_and_reverse(rnn_output, data_seq_lengths)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        
        # store all z's in here
        z_list = []
        z_mu_list = []
        z_logvar_list = []

        for t in xrange(1, T_max + 1):
            # the next two lines assemble the distribution q(z_t|z_{t-1},x_{t:T})
            z_mu, z_logvar = self.combiner(z_prev, rnn_output[:, t - 1, :])
            z_t = self.reparameterize(z_mu, z_logvar)
            z_list.append(z_t)
            z_mu_list.append(z_mu)
            z_logvar_list.append(z_logvar)
            # the latent sampled at this time step will be conditioned upon in the next time step
            # so keep track of it
            z_prev = z_t

        # list of length T w/ each element being size batch_size x z_dim
        return z_list, z_mu_list, z_logvar_list

    # define a prior network over p(z_{t+1}|z_t)
    def prior_network(self, batch_size, T_max):
        z_list = []
        z_mu_list = []
        z_logvar_list = []
        z_prev = self.z_0.expand(batch_size, self.z_0.size(0))

        for t in xrange(1, T_max + 1):
            z_mu, z_logvar = self.trans(z_prev)
            z_t = self.reparameterize(z_mu, z_logvar)

            z_list.append(z_t)
            z_mu_list.append(z_mu)
            z_logvar_list.append(z_logvar)

            z_prev = z_t

        return z_list, z_mu_list, z_logvar_list

    def forward(self, data):
        batch_size, T_max, _ = data.size()
        q_z_list, q_z_mu_list, q_z_logvar_list = self.inference_network(data)
        p_z_list, p_z_mu_list, p_z_logvar_list = self.prior_network(batch_size, T_max)

        emission_probs_list = []
        for t in xrange(1, T_max + 1):
            z_t = q_z_list[t]
            # define a generative model p(x_{1:T}|z_{1:T})
            emission_probs_t = self.emitter(z_t)
            emission_probs_list.append(emission_probs_t)

        output = {
            'q_z': q_z_list,
            'q_z_mu': q_z_mu_list,
            'q_z_logvar': q_z_logvar_list,
            'p_z': p_z_list,
            'p_z_mu': p_z_mu_list,
            'p_z_logvar': p_z_logvar_list,
            'emission_probs': emission_probs_list,
            'T_max': T_max,
        }
        
        return output


# this function takes the hidden state as output by the PyTorch rnn and
# unpacks it it; it also reverses each sequence temporally
def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences_torch(rnn_output, seq_lengths)
    return reversed_output


# this function takes a torch mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1)
# in contrast to `reverse_sequences_numpy`, this function plays
# nice with torch autograd
def reverse_sequences_torch(mini_batch, seq_lengths):
    reversed_mini_batch = mini_batch.new_zeros(mini_batch.size())
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = np.arange(T - 1, -1, -1)
        time_slice = torch.cuda.LongTensor(time_slice) if 'cuda' in mini_batch.data.type() \
            else torch.LongTensor(time_slice)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch
