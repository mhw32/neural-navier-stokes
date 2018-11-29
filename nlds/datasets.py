from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np


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
    for n in xrange(N):
        x_n = sample_lorenz_attractor(dt=dt, stepCnt=T)
        c_n = np.random.randn((100, 3))
        d_n = np.random.randn(100)
        y_n = np.dot(c_n, x_n) + d_n
        X.append(x_n)
        Y.append(y_n)
        C.append(c_n)
        D.append(d_n)
    
    X = np.vstack(X)
    Y = np.vstack(Y)
    C = np.vstack(C)
    D = np.vstack(D)

    return X, Y, C, D


def sample_lorenz_attractor(dt=0.01, stepCnt=10000):
    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))

    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    def lorenz(x, y, z, s=10, r=28, b=2.667):
        # known dynamics with fixed constants
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    data = np.concatenate([xs, ys, zs]).T
    
    return data
