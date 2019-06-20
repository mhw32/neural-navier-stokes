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
from src.spiral.utils import AverageMeter


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

    # generate toy spiral data
    print('-- creating dataset')
    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=1000,
        start=0.,
        stop=6 * np.pi,
        noise_std=.3,
        a=0., b=.3
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)

    print('-- initializing model')
    func = LatentODEfunc(4, 20).to(device)
    rec = RecognitionRNN(4, 2, 25, 1000).to(device)
    dec = Decoder(4, 2, 20).to(device)

    params = (list(func.parameters()) + list(dec.parameters()) + 
              list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    
    loss_meter = AverageMeter()

    print('-- begin training')
    for itr in tqdm(range(1, args.niters + 1)):
        optimizer.zero_grad()
        # backward in time to infer q(z_0)
        h = rec.initHidden().to(device)
        for t in reversed(range(samp_trajs.size(1))):
            obs = samp_trajs[:, t, :]
            out, h = rec.forward(obs, h)
        
        # reparameterize
        qz0_mean, qz0_logvar = out[:, :4], out[:, 4:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
        pred_x = dec(pred_z)

        # compute loss
        noise_std_ = torch.zeros(pred_x.size()).to(device) + .3
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        logpx = log_normal_pdf(
            samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

        print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

    # with torch.no_grad():
    #     # sample from trajectorys' approx. posterior
    #     h = rec.initHidden().to(device)
    #     for t in reversed(range(samp_trajs.size(1))):
    #         obs = samp_trajs[:, t, :]
    #         out, h = rec.forward(obs, h)
    #     qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
    #     epsilon = torch.randn(qz0_mean.size()).to(device)
    #     z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
    #     orig_ts = torch.from_numpy(orig_ts).float().to(device)

    #     # take first trajectory for visualization
    #     z0 = z0[0]

    #     ts_pos = np.linspace(0., 2. * np.pi, num=2000)
    #     ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()
    #     ts_pos = torch.from_numpy(ts_pos).float().to(device)
    #     ts_neg = torch.from_numpy(ts_neg).float().to(device)

    #     zs_pos = odeint(func, z0, ts_pos)
    #     zs_neg = odeint(func, z0, ts_neg)

    #     xs_pos = dec(zs_pos)
    #     xs_neg = torch.flip(dec(zs_neg), dims=[0])

    # xs_pos = xs_pos.cpu().numpy()
    # xs_neg = xs_neg.cpu().numpy()
    # orig_traj = orig_trajs[0].cpu().numpy()
    # samp_traj = samp_trajs[0].cpu().numpy()

    # plt.figure()
    # plt.plot(orig_traj[:, 0], orig_traj[:, 1],
    #         'g', label='true trajectory')
    # plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r',
    #         label='learned trajectory (t>0)')
    # plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c',
    #         label='learned trajectory (t<0)')
    # plt.scatter(samp_traj[:, 0], samp_traj[
    #             :, 1], label='sampled data', s=3)
    # plt.legend()
    # plt.savefig('./vis.png', dpi=500)
    # print('Saved visualization figure at {}'.format('./vis.png'))
