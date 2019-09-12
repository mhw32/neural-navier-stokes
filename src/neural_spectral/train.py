import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint_adjoint as odeint


class SpectralCoeffODEFunc(nn.Module):

    def __init__(self, Nx, Ny, hidden_dim=256):
        super().__init__()

        self.Nx, self.Ny = Nx, Ny
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.Nx * self.Ny * 3, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.Nx * self.Ny * 3),
        )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_path', type=str, help='where dataset is stored')
    parser.add_argument('--batch-time', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--niters', type=int, default=100)
    parser.add_argument('--gpu-device', type=int, default=0)
    args = parser.parse_args()

    device = (torch.device('cuda:' + str(args.gpu)
              if torch.cuda.is_available() else 'cpu'))

    data = np.load(args.npz_path)
    u, v, p = data['u'], data['v'], data['p']
    u = torch.from_numpy(u)
    v = torch.from_numpy(v)
    p = torch.from_numpy(p)
    x = torch.stack([u, v, p])
    x = x.to(device)
    nt, nx, ny = x.size(1), x.size(2), x.size(3)
    t = torch.arange(nt)
    t = t.to(device)

    func = SpectralCoeffODEFunc(nx, ny)
    func = func.to(device)
    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_meter = RunningAverageMeter(0.97)

    def get_batch():
        s = np.random.choice(np.arange(nt - args.batch_time, dtype=np.int64),
                             args.batch_size, replace=False)
        s = torch.from_numpy(s)
        batch_x0 = x[s]
        batch_t = t[:args.batch_time]
        batch_x = torch.stack([batch_x0[s+i] for i in range(args.batch_time)], dim=0)
        return batch_x0, batch_t, batch_x

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_x0, batch_t, batch_x = get_batch()
        # TODO.
