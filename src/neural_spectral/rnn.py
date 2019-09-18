import os
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

from src.constants import CHORIN_FD_DATA_FILE, DIRECT_FD_DATA_FILE
from src.neural_spectral.train import RunningAverageMeter, log_normal_pdf


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, obs_seq, obs_len):
        batch_size = obs_seq.size(0)

        if batch_size > 1:
            sorted_len, sorted_idx = torch.sort(obs_len, descending=True)
            obs_seq = obs_seq[sorted_idx]

        packed_len = (  sorted_len.detach().tolist() if batch_size > 1
                        else obs_len.detach().tolist()  )
        packed = rnn_utils.pack_padded_sequence(obs_seq, packed_len, 
                                                batch_first=True)
        out_seq, _ = self.gru(packed, None)
        return out_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-path', type=str, default=CHORIN_FD_DATA_FILE, 
                        help='where dataset is stored [default: CHORIN_FD_DATA_FILE]')
    parser.add_argument('--out-dir', type=str, default='./checkpoints', 
                        help='where to save checkpoints [default: ./checkpoints]')
    parser.add_argument('--n-coeff', type=int, default=10, help='default: 10')
    parser.add_argument('--batch-time', type=int, default=20, help='default: 20')
    parser.add_argument('--batch-size', type=int, default=64, help='default: 64')
    parser.add_argument('--n-iters', type=int, default=10000, help='default: 10000')
    parser.add_argument('--gpu-device', type=int, default=0, help='default: 0')
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
    
    rnn_net = RNN(nx*ny*3).to(device)
    optimizer = optim.Adam(rnn_net.parameters(), lr=1e-3)

    loss_meter = RunningAverageMeter(0.97)

    def get_batch():
        s = np.random.choice(np.arange(nt - args.batch_time, dtype=np.int64),
                             args.batch_size, replace=False)
        s = torch.from_numpy(s)
        batch_x0 = obs[s]
        batch_t = t[:args.batch_time]
        batch_x = torch.stack([obs[s+i] for i in range(args.batch_time)], dim=0)
        batch_x = batch_x.permute(1, 0, 2, 3, 4)
        return batch_t, batch_x

    try:
        tqdm_batch = tqdm(total=args.n_iters, desc="[Iteration]")
        for itr in range(1, args.n_iters + 1):
            optimizer.zero_grad()
            _, batch_obs = get_batch()
            batch_size = batch_obs.size(0)

            batch_obs = batch_obs.view(batch_size, args.batch_time, nx*ny*3)
            batch_in, batch_out = batch_obs[:, 0:-1, :], batch_obs[:, 1:, :]
            batch_len = torch.ones(batch_size).int() * args.batch_time
            batch_len = batch_len.to(device)
            batch_len = batch_len - 1  # since we split into -in and -out

            batch_pred = rnn_net(batch_in, batch_len)
            noise_std_ = torch.zeros(batch_pred.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(batch_obs, pred_obs, noise_logvar).sum(-1).sum(-1)
            loss = torch.mean(-logpx, dim=0)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            tqdm_batch.set_postfix({"Loss": loss_meter.avg})
            tqdm_batch.update()
        tqdm_batch.close()
    except KeyboardInterrupt:
        torch.save({
            'rnn_net_state_dict': rnn_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': args,
        }, os.path.join(args.out_dir, 'checkpoint.pth.tar'))

    torch.save({
        'rnn_net_state_dict': rnn_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': args,
    }, os.path.join(args.out_dir, 'checkpoint.pth.tar'))
