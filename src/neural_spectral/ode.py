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
    @param hidden_dim: integer [default: 512]
                       number of hidden dimensions
    """

    def __init__(self, latent_dim, hidden_dim=512):
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


def get_bar_c_k(k, K):
    assert k >= 0
    return 2 if (k == 0 or k == K) else 1


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
                    for k in np.arange(0, K) ] ).T
    # N(k) x N(i) since this will be multiplied by 
    # the matrix of spectral coefficients (k)
    return torch.from_numpy(T).float()


def get_inv_T_matrix(N, K):
    inv_T = np.stack([np.cos(np.pi * np.arange(N) / float(N - 1))
                      for k in np.arange(K)])
    bar_c_i = np.array([get_bar_c_k(i, N) for i in range(N)])[np.newaxis, :]
    bar_c_k = np.array([get_bar_c_k(k, K) for k in range(K)])[:, np.newaxis]

    inv_T = 2 * inv_T / (bar_c_k * bar_c_i * N)
    inv_T = torch.from_numpy(inv_T).float()
    return inv_T


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
    parser.add_argument('--n-coeff', type=int, default=10, help='default: 10')
    parser.add_argument('--batch-time', type=int, default=20, help='default: 20')
    parser.add_argument('--batch-size', type=int, default=64, help='default: 64')
    parser.add_argument('--n-iters', type=int, default=10000, help='default: 10000')
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
    T = Tx @ Ty                                    # K x K
    T_inv = np.linalg.inv(T)

    def get_u_coord(_lambda):
        return T @ _lambda

    def get_v_coord(_omega):
        return T @ _omega
    
    def get_p_coord(_gamma):
        return T @ _gamma

    def get_u_coeff(U):
        return T_inv @ U

    def get_v_coeff(V):
        return T_inv @ variables
    
    def get_p_coeff(P):
        return T_inv @ P

    latent_dim = 3 * args.n_coeff**2
    obs_dim = 3 * nx * ny
    ode_net = SpectralCoeffODEFunc(obs_dim).to(device)
    
    if not args.evaluate_only:
        optimizer = optim.Adam(ode_net.parameters()), lr=1e-3)
        loss_meter = RunningAverageMeter(0.97)

        def get_batch():
            s = np.random.choice(np.arange(nt - args.batch_time, dtype=np.int64),
                                 args.batch_size, replace=False)
            s = torch.from_numpy(s)
            batch_obs0 = obs[s]
            batch_t = t[:args.batch_time]
            batch_obs = torch.stack([obs[s+i] for i in range(args.batch_time)], dim=0)
            return batch_obs0, batch_t, batch_obs

        try:
            tqdm_batch = tqdm(total=args.n_iters, desc="[Iteration]")
            for itr in range(1, args.n_iters + 1):
                optimizer.zero_grad()
                batch_size = obs.size(0)
                mb_obs0, mb_t, mb_obs = get_batch()

                mb_U0, mb_V0, mb_P0 = mb_obs0[:, :, :, 0], mb_obs0[:, :, :, 1], mb_obs0[:, :, :, 2]
                mb_U, mb_V, mb_P = mb_obs[:, :, :, :, 0], mb_obs[:, :, :, :, 1], mb_obs[:, :, :, :, 2]
                mb_lambda0, mb_omega0, mb_gamma0 = get_u_coeff(mb_U0), get_u_coeff(mb_V0), get_u_coeff(mb_P0)
                mb_lambda, mb_omega, mb_gamma = get_u_coeff(mb_U), get_u_coeff(mb_V), get_u_coeff(mb_P)

                mb_coeff0 = torch.stack([mb_lambda0, mb_omega0, mb_gamma0])
                mb_coeff = torch.stack([mb_lambda, mb_omega, mb_gamma])

                # forward in time and solve ode for reconstructions
                pred_coeff = odeint(ode_net, mb_coeff0, mb_t.float())
                pred_coeff = pred_coeff.view(args.batch_time, batch_size, n_coeff, n_coeff, 3)
                pred_lambda, pred_omega, pred_gamma = pred_coeff[:, :, :, :, 0], pred_coeff[:, :, :, :, 1], pred_coeff[:, :, :, :, 2]
                pred_U, pred_V, pred_P = get_u_coord(pred_lambda), get_v_coord(pred_omega), get_p_coord(pred_gamma)
                pred_obs = torch.cat([pred_U.unsqueeze(4), pred_V.unsqueeze(4), pred_P.unsqueeze(4)], dim=4)
                pred_obs = pred_obs.view(batch_size, args.batch_time, -1)
                noise_std_ = torch.zeros(pred_obs.size()).to(device) + noise_std
                noise_logvar = 2. * torch.log(noise_std_).to(device)

                logpx = log_normal_pdf(mb_obs, pred_obs, noise_logvar).sum(-1).sum(-1)
                loss = torch.mean(-logpx)
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
        init_obs0 = obs[0].unsqueeze(0)  
        init_U0, init_V0, init_P0 = init_obs0[:, :, :, 0], init_obs0[:, :, :, 1], init_obs0[:, :, :, 2]
        init_lambda0, init_omega0, init_gamma0 = get_u_coeff(init_U0), get_u_coeff(init_V0), get_u_coeff(init_P0)
        init_coeff0 = torch.stack([init_lambda0, init_omega0, init_gamma0])

        pred_coeff = odeint(ode_net, init_coeff0, t.float())
        pred_coeff = pred_coeff.view(1, -1, args.n_coeff, args.n_coeff, 3)
        pred_lambda, pred_omega, pred_gamma = pred_coeff[:, :, :, :, 0], pred_coeff[:, :, :, :, 1], pred_coeff[:, :, :, :, 2]
        pred_U, pred_V, pred_P = get_u_coord(pred_lambda), get_v_coord(pred_omega), get_p_coord(pred_gamma)
        pred_obs = torch.cat([pred_U.unsqueeze(4), pred_V.unsqueeze(4), pred_P.unsqueeze(4)], dim=4)

        pred_obs = pred_obs.cpu().detach().numpy()
        
    np.save(os.path.join(args.out_dir, 'extrapolation.npy'), 
            pred_obs)
