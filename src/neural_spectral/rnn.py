import os
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim))
    
    def forward(self, obs_seq):
        batch_size, T = obs_seq.size(0), obs_seq.size(1)
        out_seq, gru_hid = self.gru(obs_seq, None)
        out_seq = out_seq.contiguous().view(batch_size * T, self.hidden_dim)
        out_seq = self.linear(out_seq)
        out_seq = out_seq.view(batch_size, T, self.input_dim)
        return out_seq, gru_hid

    def extrapolate(self, obs, T_extrapolate):
        h0 = None
        out_extrapolate = []
        for t in range(T_extrapolate):
            out, h0 = self.gru(obs, h0)
            obs = self.linear(out.squeeze(1)).unsqueeze(1)
            out_extrapolate.append(obs.cpu().detach())
        out_extrapolate = torch.cat(out_extrapolate, dim=1)
        return out_extrapolate


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-path', type=str, default='../data/data_semi_implicit.npz')
    parser.add_argument('--out-dir', type=str, default='./checkpoints/spectral', 
                        help='where to save checkpoints [default: ./checkpoints/spectral]')
    parser.add_argument('--n-iters', type=int, default=1000, help='default: 1000')
    parser.add_argument('--gpu-device', type=int, default=0, help='default: 0')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    device = (torch.device('cuda:' + str(args.gpu_device)
              if torch.cuda.is_available() else 'cpu'))

    data = np.load(args.npz_path)
    u, v, p = data['u'][:100], data['v'][:100], data['p'][:100]
    u = torch.from_numpy(u).float()
    v = torch.from_numpy(v).float()
    p = torch.from_numpy(p).float()
    obs = torch.stack([u, v, p]).permute(1, 0, 2, 3).to(device)
    nt, nx, ny = obs.size(0), obs.size(2), obs.size(3)
    obs = obs.unsqueeze(0)  # add a batch size of 1

    obs = obs.view(1, nt, 3*nx*ny)
    obs_in, obs_out = obs[:, :-1], obs[:, 1:]
    
    model = RNN(nx*ny*3, 512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_meter = AverageMeter()

    tqdm_batch = tqdm(total=args.n_iters, desc="[Iteration]")
    for itr in range(1, args.n_iters + 1):
        optimizer.zero_grad()

        obs_pred, _ = model(obs_in)
        loss = torch.norm(obs_pred - obs_out, p=2)

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

        if itr % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': args,
            }, os.path.join(args.out_dir, 'checkpoint.pth.tar'))

        tqdm_batch.set_postfix({"Loss": loss_meter.avg})
        tqdm_batch.update()
    tqdm_batch.close()

    with torch.no_grad():
        data = np.load(args.npz_path)
        u, v, p = data['u'], data['v'], data['p']
        u = torch.from_numpy(u).float()
        v = torch.from_numpy(v).float()
        p = torch.from_numpy(p).float()
        obs = torch.stack([u, v, p]).permute(1, 0, 2, 3).to(device)
        nt, nx, ny = obs.size(0), obs.size(2), obs.size(3)
        obs = obs.unsqueeze(0)  # add a batch size of 1
        obs = obs.view(1, nt, 3*nx*ny)
        obs0 = obs[:, 0].unsqueeze(1)  # first timestep - shape: mb x 3 x nx x ny

        obs_extrapolate = model.extrapolate(obs0, nt)
        obs_extrapolate = obs_extrapolate[0]
        obs_extrapolate = obs_extrapolate.numpy()
        obs_extrapolate = obs_extrapolate.reshape(obs_extrapolate.shape[0], 3, nx, ny)

    np.save(os.path.join(args.out_dir, 'extrapolation.npy'), 
            obs_extrapolate)
