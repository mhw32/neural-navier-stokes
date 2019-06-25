"""
Try to solve one of samples from a numerical approximation to 
a Lorenz system using Neural ODEs.

Heavily borrowed from https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
"""

import os
import sys
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torchdiffeq import odeint, odeint_adjoint

CUR_DIR = os.path.dirname(__file__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, help='where is data stored')
    parser.add_argument('--vis-only', action='store_true', default=False)
    parser.add_argument('--adjoint', type=eval, default=False)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    return parser


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, nbatch):
        return torch.zeros(nbatch, self.nhidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    model_dir = os.path.join(CUR_DIR, 'models')
    image_dir = os.path.join(CUR_DIR, 'images')

    if not os.path.isdir(model_dir):
        os.makedir(model_dir)

    if not os.path.isdir(image_dir):
        os.makedir(image_dir)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    data_dict = np.load(args.data_file)
    t, x, y, z = data['t'], data['x'], data['y'], data['z']
    samp_trajs, samp_ts = [x, y, z], t
    n = len(t)
    
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    test_ts = copy.deepcopy(samp_ts)

    func = LatentODEfunc(4, 20).to(device)
    rec = RecognitionRNN(4, 3, 25).to(device)
    dec = Decoder(4, 3, 20).to(device)

    if not args.vis_only:
        params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
        optimizer = optim.Adam(params, lr=args.lr)
        loss_meter = RunningAverageMeter()

        best_loss = sys.maxint

        # do actual training
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden(n).to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = torch.chunk(out, 2, dim=1)
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            if args.adjoint:
                pred_z = odeint_adjoint(func, z0, samp_ts).permute(1, 0, 2)
            else:
                pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)  # hardcoded 
            logpx = log_normal_pdf(samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

            if loss.item() < best_loss:  # save best model
                best_loss = loss.item()
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'cmd_line_args': args,
                }, os.path.join(model_dir, 'model_best.pth.tar'))

    # visualization part -- load the models
    checkpoint = torch.load(os.path.join(model_dir, 'model_best.pth.tar'))
    func.load_state_dict(checkpoint['func_state_dict'])
    rec.load_state_dict(checkpoint['rec_state_dict'])
    dec.load_state_dict(checkpoint['dec_state_dict'])

    with torch.no_grad():
        h = rec.initHidden(n).to(device)
        for t in reversed(range(samp_trajs.size(1))):
            obs = samp_trajs[:, t, :]
            out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = torch.chunk(out, 2, dim=1)
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        xs_list = []
        for i in tqdm(range(100)):
            if args.adjoint:
                zs = odeint_adjoint(func, z0[i], test_ts)
            else:
                zs = odeint(func, z0[i], test_ts)
            xs = dec(zs)
            xs = xs.cpu().numpy()
            xs_list.append(xs[np.newaxis, ...])
        xs_list = np.concatenate(xs, axis=0)

    fig, axes = plt.subplots(10, 10, figsize=(30, 30), projection='3d')
    for i in range(10):
        for j in range(10):
            index = 10 * i + j
            axes[i][j].plot(xs[index][:, 0], xs[index][:, 1], xs[index][:, 2], '-')
    plt.savefig(os.path.join(image_dir, 'vis_n_{}.pdf'.format(n), dpi=500)
