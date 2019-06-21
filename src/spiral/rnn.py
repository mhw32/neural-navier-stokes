import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.spiral.ode import Decoder
from src.spiral.dataset import generate_spiral2d
from src.spiral.utils import AverageMeter, log_normal_pdf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, nhidden=20, rnnhidden=25):
        super(RNN, self).__init__()
        self.lstm = nn.GRU(in_dim, rnnhidden, batch_first=True)
        self.dec = Decoder(latent_dim=rnnhidden, obs_dim=out_dim, nhidden=nhidden)
        self.rnnhidden = rnnhidden
        self.nhidden = nhidden

    def init_hiddens(self, batch_size):
        return torch.zeros(1, batch_size, self.rnnhidden)

    def forward(self, samp_trajs, samp_ts):
        n = samp_trajs.size(0)
        device = samp_trajs.device
        samp_ts = samp_ts.unsqueeze(0).unsqueeze(2)
        samp_ts = samp_ts.repeat(n, 1, 1)
        inputs = torch.cat((samp_trajs, samp_ts), dim=2)
        hiddens = self.init_hiddens(n)
        hiddens = hiddens.to(device)
        out, hiddens = self.lstm(inputs, hiddens)
        pred_x = self.dec(out)
        return pred_x, hiddens

    def compute_loss(self, samp_trajs, pred_x):
        device = pred_x.device
        noise_std_ = torch.zeros(pred_x.size()).to(device) + .3  # hardcoded logvar
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        logpx = log_normal_pdf(samp_trajs, pred_x, noise_logvar)
        logpx = logpx.sum(-1).sum(-1)
        loss = torch.mean(-logpx, dim=0)
        return loss

    def generate(self, samp_trajs, hiddens):
        pass


def visualize(rnn, orig_trajs, orig_ts, samp_trajs, samp_ts):
    with torch.no_grad():
        device = samp_trajs.device
        orig_ts = torch.from_numpy(orig_ts).float().to(device)
        pred_trajs = rnn(orig_trajs, orig_ts)

    orig_traj = orig_trajs[0].cpu().numpy()
    samp_traj = samp_trajs[0].cpu().numpy()

    plt.figure()
    plt.plot(orig_traj[:, 0], orig_traj[:, 1], 'g', label='true trajectory')
    plt.scatter(samp_traj[:, 0], samp_traj[:, 1], label='sampled data', s=3)
    plt.legend()
    plt.savefig('./vis.png', dpi=500)
    print('Saved visualization figure at {}'.format('./vis.png'))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='./')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=1000, start=0., stop=6 * np.pi, noise_std=.3, a=0., b=.3)
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)

    rnn = RNN(3, 2, 20, 25)
    rnn = rnn.to(device)
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr)

    loss_meter = AverageMeter()
    tqdm_pbar = tqdm(total=args.niters)
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_x, _ = rnn(samp_trajs, samp_ts)
        loss = rnn.compute_loss(samp_trajs, pred_x)
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
        'state_dict': rnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'orig_trajs': orig_trajs,
        'samp_trajs': samp_trajs,
        'orig_ts': orig_ts,
        'samp_ts': samp_ts,
    }, checkpoint_path)
