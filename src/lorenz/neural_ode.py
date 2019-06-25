"""
Try to solve one of samples from a numerical approximation to 
a Lorenz system using Neural ODEs.

Heavily borrowed from https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
"""

import os
import copy
import argparse
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
    parser.add_argument('--batch-time', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--vis-only', action='store_true', default=False)
    parser.add_argument('--adjoint', action='store_true', default=False)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--gpu', type=int, default=0)
    return parser


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3))

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


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
        os.makedirs(model_dir)

    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    data = np.load(args.data_file)
    t, x, y, z = data['t'], data['x'], data['y'], data['z']
    data = torch.from_numpy(np.vstack([x, y, z]).T)
    data0 = data[0].unsqueeze(0)
    data0 = data0.to(device )
    t = torch.from_numpy(t)
    data_size = len(t)

    def get_batch():
        s = np.random.choice(np.arange(data_size - args.batch_time, dtype=np.int64),
                             args.batch_size, replace=False))
        s = torch.from_numpy(s)
        batch_data0 = data[s]
        batch_t = t[:args.batch_time]  # (T)
        batch_data = torch.stack([data[s + i] for i in range(args.batch_time)], dim=0)
        return batch_data0, batch_t, batch_data

    func = ODEFunc()
    func = func.to(device)
    
    if not args.vis_only:
        optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        loss_meter = RunningAverageMeter(0.97)

        best_loss = np.inf
        tqdm_pbar = tqdm(total=args.niters)
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            batch_data0, batch_t, batch_data = get_batch()
            batch_data0 = batch_data0.to(device)
            batch_data = batch_data.to(device)
            batch_t = batch_t.to(device)
            pred_data = odeint(func, batch_data0, batch_t)
            train_loss = torch.mean(torch.abs(pred_data - batch_data))
            train_loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            tqdm_pbar.set_postfix({"loss": loss_meter.avg})

            if itr % 100 == 0:
                pred_data = odeint(func, data0, t)  # full thing
                test_loss = torch.mean(torch.abs(pred_data - data))
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'cmd_line_args': args,
                        'test_loss': best_loss,
                    }, os.path.join(model_dir, 'model_best_T_{}.pth.tar'.format(T)))

            tqdm_pbar.update()
        tqdm_pbar.close()

    # visualization part -- load the models
    checkpoint = torch.load(os.path.join(model_dir, 'model_best_T_{}.pth.tar'.format(T)))
    func.load_state_dict(checkpoint['func_state_dict'])

    pred_data = odeint(func, data0, t)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(pred_data[:, 0], pred_data[:, 1], pred_data[:, 2])
    plt.savefig(os.path.join(image_dir, 'vis_T_{}.pdf'.format(T)))
