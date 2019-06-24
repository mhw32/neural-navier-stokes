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
import torch.distributions as dist

from src.spiral.dataset import generate_spiral2d
from src.spiral.ldm import LDM, reverse_sequences_torch, merge_inputs
from src.spiral.utils import (AverageMeter, log_normal_pdf, normal_kl, gumbel_softmax,
                              log_mixture_of_normals_pdf, log_gumbel_softmax_pdf)


class SLDM(nn.Module):
    """
    Switching-State Linear Dynamical Model parameterizes by neural networks.

    We will only approximately be switching state by doing Gumbel-Softmax 
    with a 1 dimensional categorical variable with n_states.

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
        super().__init__()
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
            LDM(y_dim, x_dim, x_emission_dim, x_transition_dim,
                y_rnn_dim, rnn_dropout_rate=y_rnn_dropout_rate)
            for _ in range(n_states)
        ])

    def gumbel_softmax_reparameterize(self, logits, temperature, hard=False):
        """Pathwise derivatives through a soft approximation for 
        Categorical variables (i.e. Gumbel-Softmax).

        Args
        ----
        logits   := torch.Tensor
                    pre-softmax vectors for Categorical
        temperature := "softness" hyperparameter
                       for Gumbel-Softmax
        hard := boolean [default: False]
                if True, return hard sample (all weight in one class)
        """
        temperature = torch.Tensor([temperature])[0].to(logits.device)
        posterior = dist.RelaxedOneHotCategorical(temperature, logits=logits)
        return posterior.rsample()

    def gaussian_reparameterize(self, mu, logvar):
        """Pathwise derivatives through Gaussian distribution.

        Args
        ----
        mu     := torch.Tensor
                  means of Gaussian distribution
        logvar := torch.Tensor
                  diagonal log variances of Gaussian distributions
        """
        sigma = 0.5 * torch.exp(logvar)
        posterior = dist.Normal(mu, sigma)
        return posterior.rsample()

    def inference_network(self, data, temperature):
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

        Note: we should slowly decay temperature from high to low (where a low
              temperature will be closer to one-hotted.
        """
        batch_size = data.size(0)
        T = data.size(1)
        device = data.device

        # go though each system and get q(x_1:T|y_1:T)
        q_x_1_to_K, q_x_mu_1_to_K, q_x_logvar_1_to_K = [], [], []
        x_summary_1_to_K = []

        for i in range(self.n_states):
            system_i = self.systems[i]
            q_x, q_x_mu, q_x_logvar = system_i.inference_network(data)
            q_x_reversed = system_i.reverse_data(q_x)
            q_x_1_to_K.append(q_x.unsqueeze(1))
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
        q_x_1_to_K = torch.cat(q_x_1_to_K, dim=1)
        q_x_mu_1_to_K = torch.cat(q_x_mu_1_to_K, dim=1)
        q_x_logvar_1_to_K = torch.cat(q_x_logvar_1_to_K, dim=1)
        x_summary_1_to_K = torch.cat(x_summary_1_to_K, dim=2)  # batch_size x T x x_rnn_dim*n_states
        
        # start z_prev from learned initialization z_q_0
        z_prev_logits = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        z_prev = self.gumbel_softmax_reparameterize(z_prev_logits, temperature)

        x_sample, z_sample, z_logits_1_to_T = [], [], []
        for t in range(1, T + 1):
            x_summary = self.state_downsampler(x_summary_1_to_K[:, t - 1, :])
            # infer q(z_t|z_t-1,x^1_1:T,...,x^T_1:T)
            z_t_logits = self.state_combiner(z_prev, x_summary)
            # batch_size x z_dim x n_states
            z_t = self.gumbel_softmax_reparameterize(z_t_logits, temperature)
            z_prev = z_t  # update for next iter
            
            z_sample.append(z_t)
            z_logits_1_to_T.append(z_t_logits)

        z_sample_1_to_T = torch.stack(z_sample).permute(1, 0, 2)  # batch_size x T x n_states
        z_logits_1_to_T = torch.stack(z_logits_1_to_T).permute(1, 0, 2)

        return (q_x_1_to_K, q_x_mu_1_to_K, q_x_logvar_1_to_K, 
                z_sample_1_to_T, z_logits_1_to_T)

    def prior_network(self, batch_size, T, temperature):
        """Prior network for defining p(z_t | z_{t-1}). Note, unlike rsldm, this
        is independent on x!
        
        Args
        ----
        batch_size := integer
                      number of elements in a minibatch
        T := integer
             number of timesteps
        temperature := integer
                       hyperparameter for Gumbel-Softmax
        """
        # prior distribution over p(z_t|z_t-1)
        z_logits = self.z_0.expand(batch_size, self.z_0.size(0))
        z_prev = self.gumbel_softmax_reparameterize(z_logits, temperature)

        z_logits_T = []
        for t in range(1, T + 1):
            z_logits = self.state_transistor(z_prev)
            z_t = self.gumbel_softmax_reparameterize(z_logits, temperature)
            z_logits_T.append(z_logits)
            z_prev = z_t

        z_logits_T = torch.stack(z_logits_T).permute(1, 0, 2)
        return z_logits_T

    def forward(self, data, temperature):
        batch_size, T, _ = data.size()
        q_x, q_x_mu, q_x_logvar, q_z, q_z_logits = self.inference_network(data, temperature)
        p_z_logits = self.prior_network(batch_size, T, temperature)

        y_emission_mu, x_emission_mu, x_emission_logvar = [], [], []

        for t in range(1, T + 1):
            # NOTE: important to use samples from q(z,x|y)
            z_t = q_z[:, t - 1]
            x_emission_mu_t, x_emission_logvar_t = self.state_emitter(z_t)
            x_emission_mu.append(x_emission_mu_t)
            x_emission_logvar.append(x_emission_logvar_t)

            y_emission_mu_t = []
            for i in range(self.n_states):
                x_t = q_x[:, i, t - 1, :]  # q_x has shape batch_size x n_states x T x dims
                # batch_size x T x y_dim
                y_emission_mu_t_state_i = self.systems[i].emitter(x_t)
                y_emission_mu_t.append(y_emission_mu_t_state_i)
            y_emission_mu_t = torch.stack(y_emission_mu_t)
            y_emission_mu_t = y_emission_mu_t.permute(1, 0, 2)  # batch_size x n_states x dims
            y_emission_mu.append(y_emission_mu_t.unsqueeze(2))

        x_emission_mu = torch.stack(x_emission_mu).permute(1, 0, 2)
        x_emission_logvar = torch.stack(x_emission_logvar).permute(1, 0, 2)
        y_emission_mu = torch.cat(y_emission_mu, dim=2)  # batch_size x n_states x T x dims

        output = {'x_emission_mu': x_emission_mu, 'x_emission_logvar': x_emission_logvar,
                  'y_emission_mu_1_to_K': y_emission_mu, 'q_x_1_to_K': q_x, 
                  'q_x_mu_1_to_K': q_x_mu, 'q_x_logvar_1_to_K': q_x_logvar, 
                  'q_z': q_z, 'q_z_logits': q_z_logits, 'p_z_logits': p_z_logits}
        return output

    def compute_loss(self, data, output, temperature):
        T, device = data.size(1), data.device

        # fixed standard deviation in the output dimension
        noise_std_ = torch.zeros(output['y_emission_mu_1_to_K'][:, 0].size()).to(device) + .3
        noise_logvar = 2. * torch.log(noise_std_)  # hardcoded logvar

        elbo = 0
        for t in range(1, T + 1):
            elbo_t = []
            for k in range(self.n_states):
                log_p_yt_given_xt = log_normal_pdf(data[:, t - 1, :], 
                                                   output['y_emission_mu_1_to_K'][:, k, t - 1, :], 
                                                   noise_logvar[:, t - 1, :])
                log_p_xt_given_zt = log_normal_pdf(output['q_x_1_to_K'][:, k, t - 1, :],
                                                   output['x_emission_mu'][:, t - 1, :],
                                                   output['x_emission_logvar'][:, t - 1, :])
                log_p_zt_given_zt1 = log_gumbel_softmax_pdf(output['q_z'][:, t - 1, :],
                                                            output['p_z_logits'][:, t - 1, :],
                                                            temperature)
                log_q_xt_given_xt1_y = log_normal_pdf(output['q_x_1_to_K'][:, k, t - 1, :], 
                                                      output['q_x_mu_1_to_K'][:, k, t - 1, :],
                                                      output['q_x_logvar_1_to_K'][:, k, t - 1, :])
                log_q_zt_given_zt1_x1toK = log_gumbel_softmax_pdf(output['q_z'][:, t - 1, :],
                                                                  output['q_z_logits'][:, t - 1, :],
                                                                  temperature)
                elbo_t_k = log_p_yt_given_xt.sum(1) + log_p_xt_given_zt.sum(1) + log_p_zt_given_zt1 \
                             - log_q_xt_given_xt1_y.sum(1) - log_q_zt_given_zt1_x1toK
                elbo_t.append(elbo_t_k)
            elbo_t = torch.stack(elbo_t).permute(1, 0)
            state_weights = output['q_z'][:, t - 1, :]
            elbo_t = torch.sum(elbo_t * state_weights, dim=1)
            elbo += elbo_t

        elbo = torch.mean(elbo)  # across batch_size
        return -elbo


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
        self.lin_hidden_to_logits = nn.Linear(transition_dim, z_dim)

    def forward(self, z_t_1):
        h1 = self.lin_x_to_hidden(z_t_1)
        logits = self.lin_hidden_to_logits(h1)
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
        self.lin_hidden_to_logits = nn.Linear(rnn_dim, z_dim)

    def forward(self, z_t_1, h_rnn):
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = self.lin_z_to_hidden(z_t_1) + h_rnn
        # use the combined hidden state to compute the mean used to sample z_t
        z_t_logits = self.lin_hidden_to_logits(h_combined)
        return z_t_logits


class StateDownsampler(nn.Module):
    """Downsample f(x^1_{t:T}, ..., x^K_{t:T}) to a reasonable size."""
    def __init__(self, rnn_dim, n_states):
        super().__init__()
        self.lin = nn.Linear(rnn_dim * n_states, rnn_dim)
    
    def forward(self, x):
        return self.lin(x)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='./')
    return parser


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

    sldm = SLDM(2, 3, 4, 20, 20, 20, 20, 25, 25).to(device)
    optimizer = optim.Adam(sldm.parameters(), lr=args.lr)
    
    init_temp, min_temp, anneal_rate = 1.0, 0.5, 0.00003

    loss_meter = AverageMeter()
    tqdm_pbar = tqdm(total=args.niters)
    temp = init_temp
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        inputs = merge_inputs(samp_trajs, samp_ts)
        outputs = sldm(inputs, temp)
        loss = sldm.compute_loss(inputs, outputs, temp)
        if np.isnan(loss.item()):
            breakpoint();
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
        'state_dict': sldm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'orig_trajs': orig_trajs,
        'samp_trajs': samp_trajs,
        'orig_ts': orig_ts,
        'samp_ts': samp_ts,
        'temp': temp,
        'model_name': 'sldm',
    }, checkpoint_path)
