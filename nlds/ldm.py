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
        h1 = self.lin_x_to_hidden(x_t)
        #h2 = self.relu(self.lin_hidden_to_hidden(h1))
        #ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        ps = torch.sigmoid(self.lin_hidden_to_input(h1))
        return ps


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


class LDM(nn.Module):
    def __init__(
        self,
        y_dim,
        x_dim,
        x_emission_dim,
        x_transition_dim,
        rnn_dim,
        rnn_dropout_rate=0.0
    ):
        super(LDM, self).__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.x_emission_dim = x_emission_dim
        self.x_transition_dim = x_transition_dim
        self.rnn_dim = rnn_dim
        self.rnn_dropout_rate = rnn_dropout_rate

        # Get X p(x_t | x_{t-1}), TODO
        self.trans = nn.Linear(self.x_dim, self.x_transition_dim)
        # Get Y
        self.emitter = BernoulliEmitter(self.y_dim, self.x_dim, self.x_emission_dim)

        self.combiner = GaussianCombiner(self.x_dim, self.rnn_dim)
        # Define RNN over Y
        self.rnn = nn.RNN(
            input_size=self.y_dim,
            hidden_size=self.rnn_dim,
            nonlinearity='relu',
            batch_first=True,
            bidirectional=False,
            num_layers=1,
            dropout=self.rnn_dropout_rate
        )

        # Define a (trainable) parameters x_0 and x_q_0 that help define the probability
        # distributions p(x_1) and q(x_1|y_{1:T})
        # (since for t = 1 there are no previous latents to condition on)
        self.x_0 = nn.Parameter(torch.zeros(self.x_dim))
        self.x_q_0 = nn.Parameter(torch.zeros(self.x_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))

    def reverse_data(self, data):
        data_npy = data.cpu().detach().numpy()
        data_reversed_npy = data_npy[:, ::-1, :]
        # important to copy as pytorch does not work with negative numpy strides
        data_reversed = torch.from_numpy(data_reversed_npy.copy())
        data_reversed = data_reversed.to(data.device)
        return data_reversed

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # Define an inference network q(x_{1:T} | y_{1:T}) where T is maximum length
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

        h_0_contig = self.h_0.expand(1, data.size(0), self.rnn.hidden_size).contiguous()
        # push the observed y's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(data_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
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

    # Define a generative_model over p(x_t | x_{t-1})
    def generative_model(self, batch_size, T):
        x_sample_T, x_mu_T, x_logvar_T = [], [], []
        x_prev = self.x_0.expand(batch_size, self.x_0.size(0))

        for t in xrange(1, T + 1):
            x_mu = self.trans(x_prev) # Linear transform of previous state
            x_logvar = torch.zeros(x_mu.size()) # Set variance to 0 for now TODO
            x_logvar = x_logvar.to(x_mu.device)
            x_t = self.reparameterize(x_mu, x_logvar)

            x_sample_T.append(x_t)
            x_mu_T.append(x_mu)
            x_logvar_T.append(x_logvar)

            x_prev = x_t

        x_sample_T = torch.stack(x_sample_T).permute(1, 0, 2) # (batch_size, T, x_dim)
        x_mu_T = torch.stack(x_mu_T).permute(1, 0, 2)
        x_logvar_T = torch.stack(x_logvar_T).permute(1, 0, 2)

        return x_sample_T, x_mu_T, x_logvar_T


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

