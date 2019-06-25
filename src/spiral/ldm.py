"""Neural Linear Dynamical Model"""

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LDM(nn.Module):
    """
    Linear Dynamical Model parameterizes by neural networks.

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
        super().__init__()
        # p(x_t|x_t-1)
        self.transistor = Transistor(x_dim, x_transition_dim)
        # p(y_t|x_t)
        self.emitter = Emitter(y_dim, x_dim, x_emission_dim)
        # q(x_t|x_t-1,y_t:T)
        self.combiner = Combiner(x_dim, rnn_dim)
        # rnn over y
        self.rnn = nn.RNN(y_dim, rnn_dim, nonlinearity='relu', 
                          batch_first=True, dropout=rnn_dropout_rate)
        self.x_0 = nn.Parameter(torch.zeros(x_dim))    # p(x_1)
        self.x_q_0 = nn.Parameter(torch.zeros(x_dim))  # q(x_1|y_{1:T})
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

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

    def inference_network(self, data):
        # data: (batch_size, time_steps, dimension)
        batch_size = data.size(0)
        T = data.size(1)

        # flip data as RNN takes data in opposing order
        data_reversed = self.reverse_data(data)

        # compute sequence lengths
        seq_lengths = [T for _ in range(batch_size)]
        seq_lengths = np.array(seq_lengths)
        seq_lengths = torch.from_numpy(seq_lengths).long()
        seq_lengths = seq_lengths.to(data.device)

        h_0 = self.h_0.expand(1, data.size(0), self.rnn.hidden_size)
        h_0 = h_0.contiguous()

        rnn_output, _ = self.rnn(data_reversed, h_0)
        rnn_output = reverse_sequences_torch(rnn_output, seq_lengths)

        # set x_prev = x_q_0 to setup the recursive conditioning in q(x_t |...)
        x_prev = self.x_q_0.expand(batch_size, self.x_q_0.size(0))

        x_sample_T, x_mu_T, x_logvar_T = [], [], []

        for t in range(1, T + 1):
            # build q(x_t|x_{t-1},y_{t:T})
            x_mu, x_logvar = self.combiner(x_prev, rnn_output[:, t - 1, :])
            x_t = self.reparameterize(x_mu, x_logvar)

            x_sample_T.append(x_t)
            x_mu_T.append(x_mu)
            x_logvar_T.append(x_logvar)

            x_prev = x_t
        
        x_sample_T = torch.stack(x_sample_T).permute(1, 0, 2)
        x_mu_T = torch.stack(x_mu_T).permute(1, 0, 2)
        x_logvar_T = torch.stack(x_logvar_T).permute(1, 0, 2)

        return x_sample_T, x_mu_T, x_logvar_T

    def prior_network(self, batch_size, T):
        # prior distribution over p(x_t|x_t-1)
        x_prev = self.x_0.expand(batch_size, self.x_0.size(0))

        x_mu_T, x_logvar_T = [], []
        for t in range(1, T + 1):
            x_mu, x_logvar = self.transistor(x_prev)
            x_t = self.reparameterize(x_mu, x_logvar)

            x_mu_T.append(x_mu)
            x_logvar_T.append(x_logvar)

            x_prev = x_t

        x_mu_T = torch.stack(x_mu_T).permute(1, 0, 2)
        x_logvar_T = torch.stack(x_logvar_T).permute(1, 0, 2)

        return x_mu_T, x_logvar_T

    def forward(self, data):
        batch_size, T, _ = data.size()
        q_x, q_x_mu, q_x_logvar = self.inference_network(data)
        p_x_mu, p_x_logvar = self.prior_network(batch_size, T)

        y_probs = []
        for t in range(1, T + 1):
            x_t = q_x[:, t - 1]
            # define `p(y_{1:T}|x_{1:T})`  <-- this is computing the remaining term in the loss!
            y_probs_t = self.emitter(x_t)
            y_probs.append(y_probs_t)

        y_probs = torch.stack(y_probs).permute(1, 0, 2)
        output = {'q_x': q_x, 'q_x_mu': q_x_mu, 'q_x_logvar': q_x_logvar,
                  'p_x_mu': p_x_mu, 'p_x_logvar': p_x_logvar, 'y_mu': y_probs}

        return output

    def compute_loss(self, data, output):
        T = data.size(1)
        device = data.device

        # fixed standard deviation in the output dimension
        noise_std_ = torch.zeros(output['y_mu'].size()).to(device) + .3  # hardcoded logvar
        noise_logvar = 2. * torch.log(noise_std_)

        elbo = 0
        for t in range(1, T + 1):
            log_p_yt_given_xt = log_normal_pdf(data[:, t - 1, :], output['y_mu'][:, t - 1, :], noise_logvar[:, t - 1, :])
            log_p_xt_given_xt1 = log_normal_pdf(output['q_x'][:, t - 1, :],  output['p_x_mu'][:, t - 1, :],
                                                output['p_x_logvar'][:, t - 1, :])
            log_q_xt_given_xt1_y = log_normal_pdf(output['q_x'][:, t - 1, :], output['q_x_mu'][:, t - 1, :],
                                                  output['q_x_logvar'][:, t - 1, :])
            
            elbo_t = log_p_yt_given_xt.sum(1) + log_p_xt_given_xt1.sum(1) - log_q_xt_given_xt1_y.sum(1)
            elbo += elbo_t
        
        elbo = torch.mean(elbo)
        return -elbo


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
        h1 = self.lin_x_to_hidden(x_t)
        h2 = self.lin_hidden_to_hidden(h1)
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
        self.lin_hidden_to_mu = nn.Linear(transition_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(transition_dim, x_dim)
    
    def forward(self, x_t_1):
        h1 = self.lin_x_to_hidden(x_t_1)
        mu = self.lin_hidden_to_mu(h1)
        # logvar = torch.zeros_like(mu)
        logvar = self.lin_hidden_to_logvar(h1)
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
        self.lin_hidden_to_mu = nn.Linear(rnn_dim, x_dim)
        self.lin_hidden_to_logvar = nn.Linear(rnn_dim, x_dim)

    def forward(self, x_t_1, h_rnn):
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = self.lin_x_to_hidden(x_t_1) + h_rnn
        # use the combined hidden state to compute the mean used to sample z_t
        x_t_mu = self.lin_hidden_to_mu(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        # x_t_logvar = torch.zeros_like(x_t_mu) 
        x_t_logvar = self.lin_hidden_to_logvar(h_combined)
        # return parameters of normal distribution
        return x_t_mu, x_t_logvar


def reverse_sequences_torch(mini_batch, seq_lengths):
    """
    This function takes a torch mini-batch and reverses 
    each sequence w.r.t. the temporal axis, i.e. axis=1.
    """
    reversed_mini_batch = mini_batch.new_zeros(mini_batch.size())
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        time_slice = time_slice.long()
        reversed_sequence = torch.index_select(
            mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='./')
    return parser


def merge_inputs(samp_trajs, samp_ts):
    n = len(samp_trajs)
    samp_ts = samp_ts.unsqueeze(0).unsqueeze(2)
    samp_ts = samp_ts.repeat(n, 1, 1)
    inputs = torch.cat((samp_trajs, samp_ts), dim=2)
    return inputs


def visualize(ldm, orig_trajs, orig_ts, samp_trajs):
    device = orig_trajs.device
    orig_ts = torch.from_numpy(orig_ts).float().to(device)
    T = orig_ts.size(0)

    with torch.no_grad():
        # MODE 1
        # ------
        # get reconstructions with teacher forcing 
        # this requires a good inference network to be able to
        # do reconstructions.
        # inputs = merge_inputs(orig_trajs, orig_ts)
        inputs = orig_trajs
        outputs = ldm(inputs)
        recon_trajs = outputs['y_mu'][:, :, :2]  # ignore time dim

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
    plt.savefig('./vis_ldm_recons.pdf', dpi=500)

    with torch.no_grad():
        # MODE 2
        # ------
        # sample x0, use transition to get to xt, use generator to yt
        x_prev = ldm.x0.unsqueeze(0).repeat(100, ldm.x_dim)  # make 100 of x0
        x_samples, y_samples = [], []
        for t in range(T):
            x_t_mu, x_t_logvar = ldm.transistor(x_prev)
            x_t = ldm.reparameterize(x_t_mu, x_t_logvar)
            y_t_mu = ldm.emitter(x_t)
            y_t_std_ = torch.zeros_like(y_t_mu) + .3
            y_t_logvar = 2. * torch.log(y_t_std_)
            y_t = ldm.reparameterize(y_t_mu, y_t_logvar)
            x_samples.append(x_t.squeeze(1).cpu())
            y_samples.append(y_t.squeeze(1).cpu())
        x_samples = torch.cat(x_samples, dim=1)
        y_samples = torch.cat(y_samples, dim=1)
        x_samples = x_samples.numpy()
        y_samples = y_samples.numpy()

    # plot first 100 examples
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(10):
        for j in range(10):
            index = 10*i + j
            # true trajectory
            axes[i][j].plot(y_samples[index][:, 0].cpu().numpy(), 
                            y_samples[index][:, 1].cpu().numpy(), '-',
                            label='generated observations')
    axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-4, -0.12), 
                              ncol=5, fontsize=20)
    plt.savefig('./vis_ldm_y_samples.pdf', dpi=500)

    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(10):
        for j in range(10):
            index = 10*i + j
            # true trajectory
            axes[i][j].plot(x_samples[index][:, 0].cpu().numpy(), 
                            x_samples[index][:, 1].cpu().numpy(), '-',
                            label='generated latents')
    axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-4, -0.12), 
                              ncol=5, fontsize=20)
    plt.savefig('./vis_ldm_x_samples.pdf', dpi=500)


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

    ldm = LDM(2, 4, 20, 20, 25).to(device)
    # ldm = LDM(3, 4, 20, 20, 25).to(device)
    optimizer = optim.Adam(ldm.parameters(), lr=args.lr)

    loss_meter = AverageMeter()
    tqdm_pbar = tqdm(total=args.niters)
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        # inputs = merge_inputs(samp_trajs, samp_ts)
        inputs = samp_trajs
        outputs = ldm(inputs)
        loss = ldm.compute_loss(inputs, outputs)
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
        'state_dict': ldm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'orig_trajs': orig_trajs,
        'samp_trajs': samp_trajs,
        'orig_ts': orig_ts,
        'samp_ts': samp_ts,
        'model_name': 'ldm',
    }, checkpoint_path)
