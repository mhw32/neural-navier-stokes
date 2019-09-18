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
from src.neural_spectral.train import RunningAverageMeter, log_normal_pdf, normal_kl


class RNNVAE(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_dim=256):
        super().__init__()
        self.encoder = SequenceEncoder(latent_dim, obs_dim, hidden_dim=hidden_dim)
        self.decoder = SequenceDecoder(latent_dim, obs_dim, hidden_dim=hidden_dim)

    def forward(self, obs_seq_in, obs_seq_out):
        latent_mu, latent_logvar = self.encoder(obs_seq_in)
        latent = self.reparameterize(latent_mu, latent_logvar)
        obs_seq_pred = self.decoder(latent, obs_seq)
        return obs_seq_pred, latent, latent_mu, latent_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def extrapolate(self, obs_seq, T_extrapolate):
        latent_mu, latent_logvar = self.encoder(obs_seq)
        h0 = self.decoder.latent2hidden(latent_mu)
        h0 = h0.unsqueeze(0).contiguous()
        obs = obs_seq[-1]
        out_extrapolate = []
        for t in range(T_extrapolate):
            obs, h0 = self.decoder.gru(obs, h0)
            out_extrapolate.append(obs.cpu().detach())
        out_extrapolate = torch.stack(out_extrapolate)
        out_extrapolate = torch.cat([obs_seq, out_extrapolate], dim=0)
        return out_extrapolate


class SequenceEncoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.gru = nn.GRU(self.obs_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim , self.latent_dim * 2)

    def forward(self, obs_seq):
        _, hidden = self.gru(obs_seq, None)
        latent = self.linear(hidden[-1])
        mu, logvar = torch.chunk(latent, 2, dim=1)
        return mu, logvar


class SequenceDecoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(self.obs_dim, self.hidden_dim, batch_first=True)
        self.latent2hidden = nn.Linear(self.latent_dim, self.hidden_dim)

    def forward(self, latent, obs_seq):
        T = obs_seq.size(1)
        hidden = self.latent2hidden(latent)
        hidden = hidden.unsqueeze(0).contiguous()
        out_seq, _ = self.gru(obs_seq, hidden)
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
    
    # 300 is same as in train.py
    rnn_vae = RNNVAE(300, nx*ny*3).to(device)
    optimizer = optim.Adam(rnn_vae.parameters(), lr=1e-3)

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

            batch_pred, batch_latent, batch_mu, batch_logvar = rnn_vae(batch_in, batch_len)
            noise_std_ = torch.zeros(batch_obs.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)

            logpx = log_normal_pdf(batch_obs, batch_pred, noise_logvar).sum(-1).sum(-1)
            pz_mean = pz_logvar = torch.zeros(batch_latent.size()).to(device)
            analytic_kl = normal_kl(batch_mu, batch_logvar,
                                    pz_mean, pz_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            tqdm_batch.set_postfix({"Loss": loss_meter.avg})
            tqdm_batch.update()
        tqdm_batch.close()
    except KeyboardInterrupt:
        torch.save({
            'rnn_vae_state_dict': rnn_vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': args,
        }, os.path.join(args.out_dir, 'checkpoint.pth.tar'))

    torch.save({
        'rnn_vae_state_dict': rnn_vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': args,
    }, os.path.join(args.out_dir, 'checkpoint.pth.tar'))

    with torch.no_grad():
        obs_seq_init = obs[0:args.batch_time]  # give it the first batch_time seq
        obs_seq_init = obs_seq_init.unsqueeze(0)  # add fake batch size
        obs_extrapolate = rnn_vae.extrapolate(obs_seq_init, nt - args.batch_time)
        obs_extrapolate = obs_extrapolate[0]  # get rid of batch size
        obs_extrapolate = obs_extrapolate.numpy()

    np.save(os.path.join(args.out_dir, 'extrapolation.npy'), 
            obs_extrapolate)
