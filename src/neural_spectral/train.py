import os
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

from torchdiffeq import odeint
# for some reason odeint_adjoint is way too slow
# from torchdiffeq import odeint_adjoint as odeint
from src.constants import CHORIN_FD_DATA_FILE, DIRECT_FD_DATA_FILE


class SpectralCoeffODEFunc(nn.Module):
    """
    Function from a latent variable at time t to a 
    latent variable at time (t+1).

    @param latent_dim: integer
                       number of latent variables
    @param hidden_dim: integer [default: 256]
                       number of hidden dimensions
    """

    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, self.latent_dim))

    def forward(self, t, x):
        return self.net(x)


class InferenceNetwork(nn.Module):
    r"""
    Given a sequence of observations, encode them into a 
    latent variable distributed as a Gaussian distribution.

    @param latent_dim:  integer
                        number of latent variables
    @param obs_dim: integer
                    number of observed variables
    @param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """

    def __init__(self, latent_dim, obs_dim, hidden_dim=256):
        super(InferenceNetwork, self).__init__()
    
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.obs_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.latent_dim * 2)

    def forward(self, obs_seq):
        _, hidden = self.gru(obs_seq, None)
        latent = self.linear(hidden[-1])
        mu, logvar = torch.chunk(latent, 2, dim=1)
        return mu, logvar


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_gauss_lobatto_points(N, k=1):
    # N => number of points
    i = np.arange(N)
    x_i = np.cos(k * np.pi * i / float(N - 1))
    return x_i


def get_T_matrix(N, K):
    """
    Matrix of Chebyshev coefficients at collocation points.

    Matrix to convert back and forth between spectral coefficients,
    \hat{u}_k, and the values at the collocation points, u_N(x_i).
    This is just a matrix multiplication.

    \mathcal{T} = [\cos k\pi i / N], k,i = 0, ..., N
    \mathcal{U} = \mathcal{T}\hat{\mathcal{U}}

    where \mathcal{U} = [u(x_0), ..., u(x_N)], the values of the
    function at the coordinate points.
    """
    T = np.stack( [ get_gauss_lobatto_points(N, k=k)
                    for k in np.arange(K) ] ).T
    return torch.from_numpy(T).float()


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1, v2 = torch.exp(lv1), torch.exp(lv2)
    lstd1, lstd2 = lv1 / 2., lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-path', type=str, default=CHORIN_FD_DATA_FILE, 
                        help='where dataset is stored [default: CHORIN_FD_DATA_FILE]')
    parser.add_argument('--out-dir', type=str, default='./checkpoints/spectral', 
                        help='where to save checkpoints [default: ./checkpoints/spectral]')
    parser.add_argument('--batch-time', type=int, default=100, help='default: 100')
    parser.add_argument('--batch-size', type=int, default=20, help='default: 20')
    parser.add_argument('--n-iters', type=int, default=1000, help='default: 1000')
    parser.add_argument('--gpu-device', type=int, default=0, help='default: 0')
    parser.add_argument('--evaluate-only', action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    device = (torch.device('cuda:' + str(args.gpu_device)
              if torch.cuda.is_available() else 'cpu'))

    data = np.load(args.npz_path)
    u, v, p = data['u'], data['v'], data['p']
    u = torch.from_numpy(u).float()
    v = torch.from_numpy(v).float()
    p = torch.from_numpy(p).float()
    obs = torch.stack([u, v, p]).permute(1, 2, 3, 0)
    obs = obs.to(device)
    nt, nx, ny = obs.size(0), obs.size(1), obs.size(2)
    t = torch.arange(nt) + 1
    t = t.to(device)

    noise_std = 0.1
    n_coeff = 51

    # Chebyshev collocation and series
    Tx = get_T_matrix(nx, n_coeff).to(device)      # K x N
    Ty = get_T_matrix(ny, n_coeff).t().to(device)  # N x K
    T = Tx @ Ty                                         # K x K

    def build_u(_lambda):
        return T @ _lambda

    def build_v(_omega):
        return T @ _omega
    
    def build_p(_gamma):
        return T @ _gamma

    latent_dim = 3 * n_coeff**2
    obs_dim = 3 * nx * ny
    inf_net = InferenceNetwork(latent_dim, obs_dim, hidden_dim=256)
    ode_net = SpectralCoeffODEFunc(latent_dim)
    inf_net, ode_net = inf_net.to(device), ode_net.to(device)
    
    if not args.evaluate_only:
        parameters = [inf_net.parameters(), ode_net.parameters()]
        optimizer = optim.Adam(chain(*parameters), lr=1e-3)

        loss_meter = RunningAverageMeter(0.97)

        def get_batch():
            s = np.random.choice(np.arange(nt - args.batch_time, dtype=np.int64),
                                 args.batch_size, replace=False)
            s = torch.from_numpy(s)
            batch_x0 = obs[s]
            batch_t = t[:args.batch_time]
            batch_x = torch.stack([obs[s+i] for i in range(args.batch_time)], dim=0)
            batch_x = batch_x.permute(1, 0, 2, 3, 4)
            return batch_t, batch_x

        try:
            tqdm_batch = tqdm(total=args.n_iters, desc="[Iteration]")
            for itr in range(1, args.n_iters + 1):
                optimizer.zero_grad()
                batch_t, batch_obs = get_batch()
                batch_size = batch_obs.size(0)
                
                batch_obs = batch_obs.view(batch_size, args.batch_time, nx*ny*3)
                qz0_mean, qz0_logvar = inf_net(batch_obs)
                epsilon = torch.randn(qz0_mean.size()).to(device)
                pred_z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean

                # forward in time and solve ode for reconstructions
                pred_z = odeint(ode_net, pred_z0, batch_t.float())
                pred_z = pred_z.permute(1, 0, 2)  # batch_size x t x dim
                pred_z = pred_z.view(batch_size, -1, n_coeff, n_coeff, 3)
                pred_lambda = pred_z[:, :, :, :, 0]
                pred_omega  = pred_z[:, :, :, :, 1]
                pred_gamma  = pred_z[:, :, :, :, 2]
                pred_u = build_u(pred_lambda)
                pred_v = build_v(pred_omega)
                pred_p = build_p(pred_gamma)
                pred_obs = torch.cat([pred_u.unsqueeze(4), 
                                      pred_v.unsqueeze(4), 
                                      pred_p.unsqueeze(4)], dim=4)
                pred_obs = pred_obs.view(batch_size, args.batch_time, -1)
                noise_std_ = torch.zeros(pred_obs.size()).to(device) + noise_std
                noise_logvar = 2. * torch.log(noise_std_).to(device)

                logpx = log_normal_pdf(batch_obs, pred_obs, noise_logvar).sum(-1).sum(-1)
                pz0_mean = pz0_logvar = torch.zeros(pred_z0.size()).to(device)
                analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                        pz0_mean, pz0_logvar).sum(-1)
                loss = torch.mean(-logpx + analytic_kl, dim=0)
                loss.backward()
                optimizer.step()
                
                loss_meter.update(loss.item())
                tqdm_batch.set_postfix({"Loss": loss_meter.avg})
                tqdm_batch.update()
            tqdm_batch.close()
        except KeyboardInterrupt:
            torch.save({
                'ode_net_state_dict': ode_net.state_dict(),
                'inf_net_state_dict': inf_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': args,
            }, os.path.join(args.out_dir, 'checkpoint.pth.tar'))

        torch.save({
            'ode_net_state_dict': ode_net.state_dict(),
            'inf_net_state_dict': inf_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': args,
        }, os.path.join(args.out_dir, 'checkpoint.pth.tar'))
    else:
        checkpoint = torch.load(os.path.join(args.out_dir, 'checkpoint.pth.tar'))
        inf_net.load_state_dict(checkpoint['inf_net_state_dict'])
        ode_net.load_state_dict(checkpoint['ode_net_state_dict'])

    with torch.no_grad():
        obs_seq_init = obs[0:args.batch_time]  # give it the first batch_time seq
        obs_seq_init = obs_seq_init.unsqueeze(0)  # add fake batch size
        obs_seq_init = obs_seq_init.view(1, args.batch_time, -1)
        qz0_mean, qz0_logvar = inf_net(obs_seq_init)
        epsilon = torch.randn(qz0_mean.size()).to(device)
        pred_z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        t_extrapolate = torch.arange(args.batch_time, nt).to(device)
        pred_z = odeint(ode_net, pred_z0, t_extrapolate.float())
        pred_z = pred_z.permute(1, 0, 2) 
        pred_z = pred_z.view(1, -1, n_coeff, n_coeff, 3)
        pred_lambda = pred_z[:, :, :, :, 0]
        pred_omega  = pred_z[:, :, :, :, 1]
        pred_gamma  = pred_z[:, :, :, :, 2]
        pred_u = build_u(pred_lambda)
        pred_v = build_v(pred_omega)
        pred_p = build_p(pred_gamma)
        pred_obs = torch.cat([pred_u.unsqueeze(4), 
                              pred_v.unsqueeze(4), 
                              pred_p.unsqueeze(4)], dim=4)
        obs_seq_init = obs_seq_init.view(1, args.batch_time, nx, ny, 3)
        obs_extrapolate = torch.cat([obs_seq_init, pred_obs], dim=1)
        obs_extrapolate = obs_extrapolate[0]  # get rid of batch size
        obs_extrapolate = obs_extrapolate.cpu().detach().numpy()
    
    np.save(os.path.join(args.out_dir, 'extrapolation.npy'), 
            obs_extrapolate)
