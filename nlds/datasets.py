from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np
from tqdm import tqdm


class BernoulliLorenz(data.Dataset):
    r"""Wrapper around the output of build_bernoulli_lorenz_dataset"""

    def __init__(self, N, T, dt=0.01):
        self.N, self.T, self.dt = N, T, dt
        self.X, self.Y, self.C, self.D = \
            build_bernoulli_lorenz_dataset(N, T, dt=dt)
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        data = self.Y[index].T  # (T, 100)
        data = torch.from_numpy(data)

        return data


def build_bernoulli_lorenz_dataset(N, T, dt=0.01):
    r"""As in https://arxiv.org/pdf/1610.08466.pdf,
    we do not observe the Lorenz states directly. Instead,
    we simulate N=100 dimensional discrete observations
    from a generalized linear model:

        p_{t,n} = logistic(c_n^T x_t + d_n)
        y_{t,n} = Bernoulli(p_{t,n})
    
    We build a dataset upon constructing many {y_{t,n}}_{t,n}.

        x_t: 3 dim
        c_n: 100x3 dim
        d_n: 100 dim

    @param N: integer
              number of data points
    @param T: integer
              number of time steps per data point
    @param dt: float [default: 0.01]
               hyperparameter for sampling from Lorenz
    """
    X, Y, C, D = [], [], [], []
    for n in tqdm(range(N)):
        x_n = sample_lorenz_attractor(dt=dt, stepCnt=T)
        c_n = np.random.randn(100, 3)
        d_n = np.random.randn(100)
        y_n = np.dot(c_n, x_n.T) + d_n[:, np.newaxis]
        X.append(x_n)
        Y.append(y_n)
        C.append(c_n)
        D.append(d_n)
    
    X = np.rollaxis(np.dstack(X), 2, 0)
    Y = np.rollaxis(np.dstack(Y), 2, 0)
    C = np.rollaxis(np.dstack(C), 2, 0)
    D = np.rollaxis(np.dstack(D), 2, 0)

    return X, Y, C, D


def sample_lorenz_attractor(dt=0.01, stepCnt=10000):
    # Need one more for the initial values
    xs = np.empty((stepCnt,))
    ys = np.empty((stepCnt,))
    zs = np.empty((stepCnt,))

    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    def lorenz(x, y, z, s=10, r=28, b=2.667):
        # known dynamics with fixed constants
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    # Stepping through "time".
    for i in range(stepCnt - 1):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    data = np.stack([xs, ys, zs]).T
    
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='number of data points')
    parser.add_argument('T', type=int, help='number of timesteps per data point')
    parser.add_argument('--dt', type=float, default=0.01, help='hyperparameter for dynamics [default: 0.01]')
    args = parser.parse_args()

    X, Y, C, D = build_bernoulli_lorenz_dataset(args.N, args.T, dt=args.dt)
    np.save('./data/X.npy', X)
    np.save('./data/Y.npy', Y)
    np.save('./data/C.npy', C)
    np.save('./data/D.npy', D)
