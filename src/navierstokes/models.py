import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.spatial_encoder = SpatialEncoder(grid_dim, hidden_dim=hidden_dim,
                                              n_filters=n_filters)
        self.spatial_decoder = SpatialDecoder(grid_dim, hidden_dim=hidden_dim,
                                              n_filters=n_filters)
        self.rnn = nn.GRU(hidden_dim, rnn_dim, batch_first=True)
    
    def forward(self, u_seq, v_seq, p_seq):
        batch_size, T, grid_dim = u_seq.size(0), u_seq.size(1), u_seq.size(2)
        seq = torch.cat([u_seq.unsqueeze(2), v_seq.unsqueeze(2), 
                         p_seq_unsqueeze(2)], dim=2)
        seq = seq.view(batch_size * T, grid_dim, grid_dim)
        hidden_seq = self.spatial_encoder(seq)
        hidden_seq = hidden_seq.view(batch_size, T, -1)  # batch_size, T, hidden_dim
        _, hidden_seq = self.rnn(hidden_seq)
        out = self.spatial_decoder(hidden_seq)  # batch_size x channel x grid_dim**2

        next_u_seq, next_v_seq, next_p_seq = out[:, 0], out[:, 1], out[:, 2]
        return next_u_seq, next_v_seq, next_p_seq


class SpatialEncoder(nn.Module):
    def __init__(self, grid_dim, channels=3, hidden_dim=64, n_filters=32):
        super(SpatialEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters*2, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters*2, n_filters*4, 2, 2, padding=0),
            nn.ReLU())
        cout = gen_conv_output_dim(grid_dim)
        self.fc = nn.Linear(n_filters*4*cout**2, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.conv(x)
        hidden = hidden.view(batch_size, -1)
        return self.fc(hidden)


class SpatialDecoder(nn.Module):
    def __init__(self, grid_dim, channels=3, hidden_dim=64, n_filters=32)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(n_filters*4, n_filters*4, 2, 2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filters*4, n_filters*2, 2, 2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filters*2, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, channels, 1, 1, padding=0))
        self.cout = gen_conv_output_dim(grid_dim)
        self.fc = nn.Linear(hidden_dim, n_filters*4*self.cout**2)
        self.grid_dim = grid_dim
        self.channels = channels
    
    def forward(self, hidden):
        batch_size = hidden.size(0)
        out = self.fc(hidden)
        out = out.view(batch_size, self.n_filters*4, 
                       self.cout, self.cout)
        logits = self.conv(out)
        logits = out.view(batch_size, self.channels, 
                          self.grid_dim, self.grid_dim)
        return logits


def gen_conv_output_dim(s):
    s = _get_conv_output_dim(s, 2, 0, 2)
    s = _get_conv_output_dim(s, 2, 0, 2)
    s = _get_conv_output_dim(s, 2, 0, 2)
    return s


def _get_conv_output_dim(I, K, P, S):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2*P)/float(S) + 1
    return int(O)
