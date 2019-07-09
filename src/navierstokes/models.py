import torch
import torch.nn as nn
import torch.nn.functional as F

# pip install git+https://github.com/rtqichen/torchdiffeq
from torchdiffeq import odeint

# ----------------------------------------------------------------------
# Implementation of a Deterministic RNN

class RNNDiffEq(nn.Module):
    """
    Supervised approach to predict next element in a 
    differential equation.

    Args (in Forward)
    ----
    x_seq: torch.Tensor (size: batch_size x T x grid_dim x grid_dim)
           the input 
    """
    def __init__(self, grid_dim, rnn_dim=64, hidden_dim=64, n_filters=32):
        super(RNNDiffEq, self).__init__()
        self.bc_encoder = BoundaryConditionEncoder(grid_dim, rnn_dim, channels=channels,
                                                   hidden_dim=hidden_dim, n_filters=n_filters)
        self.spatial_encoder = SpatialEncoder(grid_dim, hidden_dim=hidden_dim,
                                              n_filters=n_filters)
        self.spatial_decoder = SpatialDecoder(grid_dim, hidden_dim=hidden_dim,
                                              n_filters=n_filters)
        self.rnn = nn.GRU(hidden_dim, rnn_dim, batch_first=True)
    
    def forward(self, u_seq, v_seq, p_seq, rnn_h0=None):
        batch_size, T, grid_dim = u_seq.size(0), u_seq.size(1), u_seq.size(2)
        seq = torch.cat([u_seq.unsqueeze(2), v_seq.unsqueeze(2), 
                         p_seq.unsqueeze(2)], dim=2)
    
        if rnn_h0 is None:
            # pull out boundary conditions (which should be constant over time)
            bc_x0, bc_xn = seq[:, 0, :, 0, :], seq[:, 0, :, -1, :]
            bc_y0, bc_yn = seq[:, 0, :, :, 0], seq[:, 0, :, :, -1]
            rnn_h0 = self.bc_encoder(bc_x0, bc_xn, bc_y0, bc_yn)

        seq = seq.view(batch_size * T, 3, grid_dim, grid_dim)
        hidden_seq = self.spatial_encoder(seq)
        hidden_seq = hidden_seq.view(batch_size, T, -1)  # batch_size, T, hidden_dim
        output_seq, rnn_h = self.rnn(hidden_seq, rnn_h0)
        output_seq = output_seq.contiguous().view(batch_size * T, -1)
        out = self.spatial_decoder(output_seq)  # batch_size x channel x grid_dim**2
        out = out.view(batch_size, T, 3, grid_dim, grid_dim)

        next_u_seq, next_v_seq, next_p_seq = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        next_u_seq = next_u_seq.contiguous()
        next_v_seq = next_v_seq.contiguous()
        next_p_seq = next_p_seq.contiguous()

        return next_u_seq, next_v_seq, next_p_seq, rnn_h


class SpatialEncoder(nn.Module):
    def __init__(self, grid_dim, channels=3, hidden_dim=64, n_filters=32):
        super(SpatialEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, n_filters, 2, 2, padding=0),
            nn.ReLU())
            # nn.Conv2d(n_filters, n_filters*2, 2, 2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(n_filters*2, n_filters*4, 2, 2, padding=0),
            # nn.ReLU())
        cout = gen_conv_output_dim(grid_dim)
        self.fc = nn.Linear(n_filters*cout**2, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.conv(x)
        hidden = hidden.view(batch_size, -1)
        return self.fc(hidden)


class SpatialDecoder(nn.Module):
    def __init__(self, grid_dim, channels=3, hidden_dim=64, n_filters=32):
        super(SpatialDecoder, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(n_filters, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            # nn.ConvTranspose2d(n_filters*4, n_filters*2, 2, 2, padding=0),
            # nn.ReLU(),
            # nn.ConvTranspose2d(n_filters*2, n_filters, 2, 2, padding=0),
            # nn.ReLU(),
            nn.Conv2d(n_filters, channels, 1, 1, padding=0))
        self.cout = gen_conv_output_dim(grid_dim)
        self.fc = nn.Linear(hidden_dim, n_filters*self.cout**2)
        self.grid_dim = grid_dim
        self.channels = channels
        self.n_filters = n_filters
    
    def forward(self, hidden):
        batch_size = hidden.size(0)
        out = self.fc(hidden)
        out = out.view(batch_size, self.n_filters, self.cout, self.cout)
        logits = self.conv(out)
        logits = logits.view(batch_size, self.channels, 
                             self.grid_dim, self.grid_dim)
        return logits


class BoundaryConditionEncoder(nn.Module):
    def __init__(self, grid_dim, out_dim, channels=3, 
                 hidden_dim=64, n_filters=32):
        self.x0_bc = BoundaryConditionNetwork(
            grid_dim, channels=channels, 
            hidden_dim=hidden_dim, n_filters=n_filters)
        self.xn_bc = BoundaryConditionNetwork(
            grid_dim, channels=channels, 
            hidden_dim=hidden_dim, n_filters=n_filters)
        self.y0_bc = BoundaryConditionNetwork(
            grid_dim, channels=channels, 
            hidden_dim=hidden_dim, n_filters=n_filters)
        self.yn_bc = BoundaryConditionNetwork(
            grid_dim, channels=channels, 
            hidden_dim=hidden_dim, n_filters=n_filters)
        self.fc = nn.Linear(hidden_dim*4, out_dim)

    def forward(self, x0, xn, y0, yn):
        h_x0 = self.x0_bc(x0)
        h_xn = self.xn_bc(xn)
        h_y0 = self.y0_bc(y0)
        h_yn = self.yn_bc(yn)
        h_bc = torch.cat([h_x0, h_xn, h_y0, h_yn], dim=1)
        return self.fc(F.relu(h_bc))


class BoundaryConditionNetwork(nn.Module):
    """
    Encode the boundary conditions as 1 dimensional
    convolutions over a single boundary.
    """
    def __init__(self, grid_dim, channels=3, 
                 hidden_dim=64, n_filters=32):
        self.boundary_encoder = nn.Sequential(
            nn.Conv1d(channels, n_filters, 2, padding=0),
            nn.ReLU())
        cout = gen_conv_output_dim(grid_dim)
        self.fc = nn.Linear(n_filters*cout, hidden_dim)
    
    def forward(self, bc):
        return self.fc(F.relu(self.boundary_encoder(bc)))


def gen_conv_output_dim(s):
    s = _get_conv_output_dim(s, 2, 0, 2)
    # s = _get_conv_output_dim(s, 2, 0, 2)
    # s = _get_conv_output_dim(s, 2, 0, 2)
    return s


def _get_conv_output_dim(I, K, P, S):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2*P)/float(S) + 1
    return int(O)

# ----------------------------------------------------------------------
# Implementation of NeuralODEs in PyTorch (adapted from TorchDiffEq)

class ODEDiffEq(nn.Module):
    """
    Latent variable model with built-in ODE to predict next element 
    in a differential equation. This model is stochastic by definition.
    Uses an RNN internally for encoding sequential data.

    Args (in Forward)
    ----
    x_seq: torch.Tensor (size: batch_size x T x grid_dim x grid_dim)
           the input 
    """
    def __init__(self, grid_dim, latent_dim=16, rnn_dim=64, hidden_dim=64, n_filters=32):
        super(ODEDiffEq, self).__init__()
        self.ode_func = LatentODEfunc(latent_dim, hidden_dim=hidden_dim)
        self.bc_encoder = BoundaryConditionEncoder(grid_dim, rnn_dim, channels=channels,
                                                   hidden_dim=hidden_dim, n_filters=n_filters)
        self.spatial_encoder = SpatialEncoder(grid_dim, hidden_dim=hidden_dim,
                                              n_filters=n_filters)
        self.spatial_decoder = SpatialDecoder(grid_dim, hidden_dim=hidden_dim,
                                              n_filters=n_filters)
        self.rnn = nn.GRU(hidden_dim, rnn_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim * 2)
        self.latent_dim = latent_dim

    def infer(self, u_seq, v_seq, p_seq, rnn_h0=None):
        batch_size, T, grid_dim = u_seq.size(0), u_seq.size(1), u_seq.size(2)
        seq = torch.cat([u_seq.unsqueeze(2), v_seq.unsqueeze(2), 
                         p_seq.unsqueeze(2)], dim=2)

        if rnn_h0 is None:
            # pull out boundary conditions (which should be constant over time)
            bc_x0, bc_xn = seq[:, 0, :, 0, :], seq[:, 0, :, -1, :]
            bc_y0, bc_yn = seq[:, 0, :, :, 0], seq[:, 0, :, :, -1]
            rnn_h0 = self.bc_encoder(bc_x0, bc_xn, bc_y0, bc_yn)

        seq = seq.view(batch_size * T, 3, grid_dim, grid_dim)
        hidden_seq = self.spatial_encoder(seq)
        hidden_seq = hidden_seq.view(batch_size, T, -1)  # batch_size, T, hidden_dim
        _, rnn_h = self.rnn(hidden_seq, rnn_h0)
        latent_chunk = self.hidden_to_latent(rnn_h.squeeze(0))
        latent_mu, latent_logvar = torch.chunk(latent_chunk, 2, dim=1)
        latent = self.reparameterize(latent_mu, latent_logvar)
        
        return latent, latent_mu, latent_logvar, rnn_h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, u_seq, v_seq, p_seq, t_seq, rnn_h0=None):
        z0, qz0_mean, qz0_logvar, rnn_h = self.infer(u_seq, v_seq, p_seq, rnn_h0=rnn_h0)
        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.ode_func, z0, t_seq).permute(1, 0, 2)
        batch_size, T, dim = pred_z.size()
        pred_z_flat = pred_z.view(batch_size * T, dim)
        pred_x_flat = self.spatial_decoder(pred_z_flat)
        pred_x = pred_x_flat.view(batch_size, T, 3, grid_dim, grid_dim)

        next_u_seq, next_v_seq, next_p_seq = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        next_u_seq = next_u_seq.contiguous()
        next_v_seq = next_v_seq.contiguous()
        next_p_seq = next_p_seq.contiguous()

        return next_u_seq, next_v_seq, next_p_seq, z0, qz0_mean, qz0_logvar, rnn_h


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.elu(self.fc1(x))
        out = self.elu(self.fc2(out))
        out = self.elu(self.fc3(out))
        out = self.fc4(out)
        return out
