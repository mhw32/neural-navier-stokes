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
        self.gru = nn.GRU(in_dim, rnnhidden, batch_first=True)
        self.dec = Decoder(latent_dim=rnnhidden, obs_dim=out_dim, nhidden=nhidden)
        self.rnnhidden = rnnhidden
        self.nhidden = nhidden

    def init_hiddens(self, batch_size):
        return torch.zeros(1, batch_size, self.rnnhidden)

    def forward(self, samp_trajs, samp_ts, hiddens=None):
        n = samp_trajs.size(0)
        device = samp_trajs.device

        samp_ts = samp_ts.unsqueeze(0).unsqueeze(2)
        samp_ts = samp_ts.repeat(n, 1, 1)
        inputs = torch.cat((samp_trajs, samp_ts), dim=2)

        if hiddens is None:
            hiddens = self.init_hiddens(n)
            hiddens = hiddens.to(device)
        
        out, hiddens = self.gru(inputs, hiddens)
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


def visualize(rnn, orig_trajs, orig_ts, samp_trajs, index=0):
    device = orig_trajs.device
    orig_ts = torch.from_numpy(orig_ts).float().to(device)
    T = orig_ts.size(0)

    with torch.no_grad():
        # MODE 1
        # ------
        # this is teacher forcing -- show a point, predict a point
        #                            this should be very easy!
        recon_trajs, _ = rnn(orig_trajs, orig_ts)
        
        # MODE 2
        # ------
        # this takes only the FIRST sample and propogates forward
        # from there -- this is very difficult!
        extra_trajs = []
        trajs = orig_trajs[:, 0, :].unsqueeze(1)  # take first sample
        hiddens = None
        for t in range(T):
            trajs, hiddens = rnn(trajs, orig_ts[t].unsqueeze(0), hiddens=hiddens) 
            extra_trajs.append(trajs)
        extra_trajs = torch.cat(extra_trajs, dim=1)

        # MODE 3
        # ------
        # Take the first 500 and then generate step by step
        # this is medium difficulty but is the most interesting
        middle_trajs_1, hiddens = rnn(orig_trajs[:, :T // 2], orig_ts[:T // 2])
        middle_trajs_2 = []
        trajs = orig_trajs[:, T // 2].unsqueeze(1)  # given true sample
        for t in range(T // 2, T):
            trajs, hiddens = rnn(trajs, orig_ts[t].unsqueeze(0), hiddens=hiddens) 
            middle_trajs_2.append(trajs)
        middle_trajs_2 = torch.cat(middle_trajs_2, dim=1)
        middle_trajs = torch.cat((middle_trajs_1, middle_trajs_2), dim=1)

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
            # learned trajectory (generated)
            axes[i][j].plot(extra_trajs[index][:, 0].cpu().numpy(), 
                            extra_trajs[index][:, 1].cpu().numpy(), '-',
                            label='generated')
            # learned trajectory (middle ground)
            axes[i][j].plot(middle_trajs[index][:, 0].cpu().numpy(), 
                            middle_trajs[index][:, 1].cpu().numpy(), '-',
                            label='half teacher forcing')
            axes[i][j].plot(samp_trajs[index][:, 0].cpu().numpy(), 
                            samp_trajs[index][:, 1].cpu().numpy(), 
                            'o', markersize=1,
                            label='dataset')
    axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-4, -0.12), 
                              ncol=5, fontsize=20)
    plt.savefig('./vis_rnn.pdf', dpi=500)


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
        'model_name': 'rnn',
    }, checkpoint_path)
