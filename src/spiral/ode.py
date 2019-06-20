import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# pip install git+https://github.com/rtqichen/torchdiffeq
from torchdiffeq import odeint
from src.spiral.dataset import generate_spiral2d
from src.spiral.utils import AverageMeter, log_normal_pdf, normal_kl


class NeuralODE(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20, rnnhidden=25, nbatch=1):
        super(NeuralODE, self).__init__()
        self.func = LatentODEfunc(latent_dim=latent_dim, nhidden=nhidden)
        self.rec = RecognitionRNN(latent_dim=latent_dim, obs_dim=obs_dim, 
                                  nhidden=rnnhidden, nbatch=nbatch)
        self.dec = Decoder(latent_dim=latent_dim, obs_dim=obs_dim, 
                           nhidden=nhidden)
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.rnnhidden = rnnhidden
        self.nbatch = nbatch
    
    def forward(self, samp_trajs, samp_ts):
        device = samp_trajs.device
        # backward in time to infer q(z_0)
        h = self.rec.initHidden().to(device)
        
        for t in reversed(range(samp_trajs.size(1))):
            obs = samp_trajs[:, t, :]
            out, h = self.rec.forward(obs, h)
        
        # reparameterize
        qz0_mean, qz0_logvar = out[:, :4], out[:, 4:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, samp_ts).permute(1, 0, 2)
        pred_x = self.dec(pred_z)

        return pred_x, z0, qz0_mean, qz0_logvar

    def compute_loss(self, pred_x, z0, qz0_mean, qz0_logvar):
        device = pred_x.device
        noise_std_ = torch.zeros(pred_x.size()).to(device) + .3  # hardcoded logvar
        noise_logvar = 2. * torch.log(noise_std_).to(device)

        logpx = log_normal_pdf(samp_trajs, pred_x, noise_logvar)
        logpx = logpx.sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        return loss


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
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
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

    ode = NeuralODE(4, 2, 20, 25, 1000).to(device)
    optimizer = optim.Adam(ode.parameters(), lr=args.lr)
    
    loss_meter = AverageMeter()
    tqdm_pbar = tqdm(total=args.niters)
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_x, z0, qz0_mean, qz0_logvar = ode(samp_trajs, samp_ts)
        loss = ode.compute_loss(pred_x, z0, qz0_mean, qz0_logvar)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        tqdm_pbar.set_postfix({"elbo": -loss_meter.avg})
        tqdm_pbar.update()
    tqdm_pbar.close()
