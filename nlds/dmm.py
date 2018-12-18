r"""Implementation is heavily borrowed from Pyro's implementation.

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag

Let x refer to a Gaussian latent variables. Let y be a Bernoulli
observed variable.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BernoulliEmitter(nn.Module):
    r"""Parameterizes the bernoulli observation likelihood `p(y_t | x_t)`
    
    @param y_dim: integer
                  number of input dimensions
    @param x_dim: integer
                  number of latent dimensions
    @param emission_dim: integer
                         number of hidden dimensions in output 
                         neural network
    """
    def __init__(self, y_dim, x_dim, emission_dim):
        super(BernoulliEmitter, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_x_to_hidden = nn.Linear(x_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, y_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, x_t):
        r"""Given the latent x at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(y_t|x_t)`
        """
        h1 = self.relu(self.lin_x_to_hidden(x_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps


class GaussianGatedTransition(nn.Module):
    r"""Parameterizes the gaussian latent transition probability `p(x_t | x_{t-1})`
    See section 5 in the reference for comparison.

    @param z_dim: integer
                  number of latent dimensions
    @param transition_dim: integer
                           number of transition dimensions
    """
    def __init__(self, x_dim, transition_dim):
        super(GaussianGatedTransition, self).__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_x_to_hidden = nn.Linear(x_dim, transition_dim)
        self.lin_gate_hidden_to_x = nn.Linear(transition_dim, x_dim)
        self.lin_proposed_mean_x_to_hidden = nn.Linear(x_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_x = nn.Linear(transition_dim, x_dim)
        self.lin_x_to_logvar = nn.Linear(x_dim, x_dim)
        self.lin_x_to_mu = nn.Linear(x_dim, x_dim)
        # modify the default initialization of lin_x_to_mu
        # so that it's starts out as the identity function
        self.lin_x_to_mu.weight.data = torch.eye(x_dim)
        self.lin_x_to_mu.bias.data = torch.zeros(x_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, x_t_1):
        r"""Given the latent `x_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(x_t | x_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_x_to_hidden(x_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_x(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_x_to_hidden(x_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_x(_proposed_mean)
        # assemble the actual mean used to sample x_t, which mixes a linear transformation
        # of x_{t-1} with the proposed mean modulated by the gating function
        x_t_mu = (1 - gate) * self.lin_x_to_mu(x_t_1) + gate * proposed_mean
        # compute the scale used to sample x_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        x_t_logvar = self.lin_x_to_logvar(self.relu(proposed_mean))
        # return parameters of normal distribution
        return x_t_mu, x_t_logvar


class GaussianCombiner(nn.Module):
    r"""Parameterizes `q(x_t | x_{t-1}, y_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `y_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)

    NOTE: x_{t-1} is supposed to capture y_{1:t-1}. So, `q(x_t|x_{t-1},y_{t:T})` 
    refers to `q(x_t|y_{1:T})`.

    @param x_dim: integer
                  number of latent dimensions
    @param rnn_dim: integer
                    hidden dimensions of RNN
    """
    def __init__(self, x_dim, rnn_dim):
        super(GaussianCombiner, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_x_to_hidden = nn.Linear(x_dim, rnn_dim)
        self.lin_hidden_to_mu = nn.Linear(rnn_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(rnn_dim, x_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()

    def forward(self, x_t_1, h_rnn):
        r"""Given the latent x at at a particular time step t-1 as well as the hidden
        state of the RNN `h(y_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(x_t | x_{t-1}, y_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_x_to_hidden(x_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        x_t_mu = self.lin_hidden_to_mu(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        x_t_logvar = self.lin_hidden_to_logvar(h_combined)
        # return parameters of normal distribution
        return x_t_mu, x_t_logvar


class DMM(nn.Module):
    r"""Deep Markov Model.
    
    NOTE: I assume all inputs are the SAME length. We need to fix this later
          but I bet this assumption will make initial development much easier.
    
    @param y_dim: integer
                  number of input dimensions 
    @param x_dim: integer
                  number of latent dimensions
    @param emissions_dim: integer
                          number of hidden dimensions in network to 
                          generate outputs
    @param transition_dim: integer
                           number of transition dimensions
    @param rnn_dim: integer
                    hidden dimensions of RNN
    @param rnn_dropout_rate: float [default: 0.0]
                             dropout rate for RNN
    """
    def __init__(self, y_dim, x_dim, emission_dim, transition_dim, 
                 rnn_dim, rnn_dropout_rate=0.0):
        super(DMM, self).__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.rnn_dropout_rate = rnn_dropout_rate

        self.emitter = BernoulliEmitter(self.y_dim, self.x_dim, self.emission_dim)
        self.trans = GaussianGatedTransition(self.x_dim, self.transition_dim)
        self.combiner = GaussianCombiner(self.x_dim, self.rnn_dim)
        self.rnn = nn.RNN(input_size=self.y_dim, hidden_size=self.rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=1,
                          dropout=self.rnn_dropout_rate)

        # define a (trainable) parameters x_0 and x_q_0 that help define the probability
        # distributions p(x_1) and q(x_1|y_{1:T})
        # (since for t = 1 there are no previous latents to condition on)
        self.x_0 = nn.Parameter(torch.zeros(self.x_dim))
        self.x_q_0 = nn.Parameter(torch.zeros(self.x_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def reverse_data(self, data):
        data_npy = data.cpu().detach().numpy()
        data_reversed_npy = data_npy[:, ::-1, :]
        # important to copy as pytorch does not work with negative numpy strides
        data_reversed = torch.from_numpy(data_reversed_npy.copy())
        data_reversed = data_reversed.to(data.device)
        return data_reversed

    # define an inference network q(x_{1:T}|y_{1:T}) where T is maximum length
    def inference_network(self, data):
        # data: (batch_size, time_steps, dimension)
        # data := y
        # this is the number of time steps we need to process in the mini-batch
        batch_size = data.size(0)
        T = data.size(1)
        
        # flip data as RNN takes data in opposing order
        data_reversed = self.reverse_data(data)

        # compute sequence lengths
        data_seq_lengths = [T for _ in xrange(batch_size)]
        data_seq_lengths = np.array(data_seq_lengths)
        data_seq_lengths = torch.from_numpy(data_seq_lengths).long()
        data_seq_lengths = data_seq_lengths.to(data.device)

        h_0_contig = self.h_0.expand(
            1, data.size(0), self.rnn.hidden_size).contiguous()
        # push the observed y's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(data_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state
        rnn_output = reverse_sequences_torch(rnn_output, data_seq_lengths)
        # set x_prev = x_q_0 to setup the recursive conditioning in q(x_t |...)
        x_prev = self.x_q_0.expand(batch_size, self.x_q_0.size(0))
        
        # store all z's in here
        x_sample_T, x_mu_T, x_logvar_T = [], [], []

        for t in xrange(1, T + 1):
            # the next two lines assemble the distribution q(x_t|x_{t-1},y_{t:T})
            x_mu, x_logvar = self.combiner(x_prev, rnn_output[:, t - 1, :])
            x_t = self.reparameterize(x_mu, x_logvar)
            
            x_sample_T.append(x_t)
            x_mu_T.append(x_mu)
            x_logvar_T.append(x_logvar)
            
            # the latent sampled at this time step will be conditioned upon in the next time step
            # so keep track of it
            x_prev = x_t

        x_sample_T = torch.stack(x_sample_T).permute(1, 0, 2)  # (batch_size, T, x_dim)
        x_mu_T = torch.stack(x_mu_T).permute(1, 0, 2)
        x_logvar_T = torch.stack(x_logvar_T).permute(1, 0, 2)

        return x_sample_T, x_mu_T, x_logvar_T

    # define a generative_model over p(x_t|x_{t-1})
    def generative_model(self, batch_size, T):
        x_sample_T, x_mu_T, x_logvar_T = [], [], []
        x_prev = self.x_0.expand(batch_size, self.x_0.size(0))

        for t in xrange(1, T + 1):
            x_mu, x_logvar = self.trans(x_prev)
            x_t = self.reparameterize(x_mu, x_logvar)

            x_sample_T.append(x_t)
            x_mu_T.append(x_mu)
            x_logvar_T.append(x_logvar)

            x_prev = x_t

        x_sample_T = torch.stack(x_sample_T).permute(1, 0, 2)  # (batch_size, T, x_dim)
        x_mu_T = torch.stack(x_mu_T).permute(1, 0, 2)
        x_logvar_T = torch.stack(x_logvar_T).permute(1, 0, 2)

        return x_sample_T, x_mu_T, x_logvar_T

    def forward(self, data):
        batch_size, T, _ = data.size()
        q_x, q_x_mu, q_x_logvar = self.inference_network(data)
        p_x, p_x_mu, p_x_logvar = self.generative_model(batch_size, T)

        y_emission_probs = []
        for t in xrange(1, T + 1):
            x_t = q_x[:, t]
            # define a generative model p(y_{1:T}|x_{1:T})
            y_emission_probs_t = self.emitter(x_t)
            y_emission_probs.append(y_emission_probs_t)

        y_emission_probs = torch.stack(y_emission_probs)
        y_emission_probs = y_emission_probs.permute(1, 0, 2)

        output = {
            'q_x': q_x,
            'q_x_mu': q_x_mu,
            'q_x_logvar': q_x_logvar,
            'p_x': p_x,
            'p_x_mu': p_x_mu,
            'p_x_logvar': p_x_logvar,
            'y_emission_probs': y_emission_probs,
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
