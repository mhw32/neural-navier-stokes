import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint


class Lambda(nn.Module):
    def __init__(self, true_A):
        super().__init__()
        self.true_A = true_A

    def forward(self, t, y):
        return torch.mm(y**3, self.true_A)


def true_ode(data_size):
    true_y0 = torch.tensor([[2., 0.]])
    t = torch.linspace(0., 25., data_size)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
    
    with torch.no_grad():
        # dopri = Runga-Kutta method
        true_y = odeint(Lambda(true_A), true_y0, t, method='dopri5')

    return true_y0, true_y, t


def gen_batch(true_y, t, data_size, batch_size, batch_time):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y
