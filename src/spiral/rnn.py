import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.spirals.ode import get_parser, Decoder
from src.spiral.dataset import generate_spiral2d
from src.spiral.utils import AverageMeter, log_normal_pdf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, obs_dim=2, nhidden=20, rnnhidden=25, max_ts=100):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(obs_dim, rnnhidden, batch_first=True)
        self.dec = Decoder(latent_dim=rnnhidden, obs_dim=obs_dim,
                           nhidden=nhidden)
        self.max_ts = max_ts

    def forward(self, samp_trajs, samp_ts):
        inputs = torch.cat((samp_trajs, samp_ts / self.max_ts), dim=1)
        out = self.lstm(inputs)
        pred_x = self.dec(out)
        return pred_x

    def compute_loss(self, samp_trajs, pred_x):
        device = pred_x.device
        noise_std_ = torch.zeros(pred_x.size()).to(device) + .3  # hardcoded logvar
        noise_logvar = 2. * torch.log(noise_std_).to(device)

        logpx = log_normal_pdf(samp_trajs, pred_x, noise_logvar)
        logpx = logpx.sum(-1).sum(-1)
        loss = torch.mean(-logpx, dim=0)
        return loss


def visualize(rnn, orig_trajs, samp_trajs, orig_ts):
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

    rnn = RNN(2, 20, 25)
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr)

    loss_meter = AverageMeter()
    tqdm_pbar = tqdm(total=args.niters)
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_x = rnn(samp_trajs, samp_ts)
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
