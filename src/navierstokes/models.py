import torch
import torch.nn as nn
import torch.nn.functional as F

# pip install git+https://github.com/rtqichen/torchdiffeq
from torchdiffeq import odeint_adjoint as odeint

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
        self.bc_encoder = BoundaryConditionEncoder(
            grid_dim, rnn_dim, hidden_dim=hidden_dim, n_filters=n_filters)
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
            rnn_h0 = rnn_h0.unsqueeze(0)

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
            nn.Conv2d(channels, n_filters, 2, 2, padding=0))
            # nn.ReLU(),
            # nn.Conv2d(n_filters, n_filters*2, 2, 2, padding=0))
        self.cout = gen_conv_output_dim(grid_dim)
        self.fc = nn.Linear(n_filters*self.cout**2, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = F.relu(self.conv(x))
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
        super().__init__()
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
        super().__init__()
        self.boundary_encoder = nn.Sequential(
            nn.Conv1d(channels, n_filters // 2, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(n_filters // 2, n_filters, 3, padding=0))
        self.fc = nn.Linear(n_filters*6, hidden_dim)
    
    def forward(self, bc):
        batch_size = bc.size(0)
        hid = F.relu(self.boundary_encoder(bc))
        hid = hid.view(batch_size, 32 * 6)
        return self.fc(hid)


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
    Supervised model with built-in ODE to predict next element 
    in a differential equation. Turns out this is almost the same
    except we need to handle the input/output scheme a little differently.
    """
    def __init__(self, grid_dim, hidden_dim=64, n_filters=32):
        super(ODEDiffEq, self).__init__()
        self.bc_encoder = BoundaryConditionEncoder(
            grid_dim, hidden_dim, hidden_dim=hidden_dim, n_filters=n_filters)
        self.spatial_encoder = SpatialEncoder(grid_dim, hidden_dim=hidden_dim,
                                              n_filters=n_filters)
        self.spatial_decoder = SpatialDecoder(grid_dim, hidden_dim=hidden_dim,
                                              n_filters=n_filters)
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim))
        self.grid_dim = grid_dim

    def forward(self, t, obs):
        batch_size = obs.size(0)

        x0_bc, xn_bc = obs[:, :, 0, :], obs[:, :, -1, :]
        y0_bc, yn_bc = obs[:, :, :, 0], obs[:, :, :, -1]
        h_bc = self.bc_encoder(x0_bc, xn_bc, y0_bc, yn_bc)

        h_obs = self.spatial_encoder(obs)
        hidden = torch.cat((h_bc, h_obs), dim=1)
        hidden = self.combiner(hidden)

        out = self.spatial_decoder(hidden)  # batch_size x channel x grid_dim**2
        out = out.view(batch_size, 3, self.grid_dim, self.grid_dim)
        return out


class ODEDiffEqElement(nn.Module):
    def __init__(self, i, j, grid_dim, hidden_dim=64, n_filters=32):
        super(ODEDiffEqElement, self).__init__()
        self.i, self.j = i, j
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3))

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0) 

    def forward(self, t, obs_ij):
        return self.net(obs_ij)


# ---------------------------------------------------------------------------
# Implementation of Residual NeuralODEs in PyTorch (adapted from TorchDiffEq)

class ResidualODEDiffEq(nn.Module):
    def __init__(self, grid_dim, n_filters=32):
        super(ResidualODEDiffEq, self).__init__()
        self.model = RDN(3, nFeat=n_filters)

    def forward(self, t, obs):
        return self.model(obs)


class RDN(nn.Module):
    def __init__(self, nChannel, nDenselayer=3, nFeat=64, scale=3, growthRate=32):
        super(RDN, self).__init__()
        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3 
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):
        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)     
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        # us = self.conv_up(FDF)
        # us = self.upsample(us)
        output = self.conv3(FDF)
        return output


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, 
                                  padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.body(x)
        return x


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, 
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
    
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
