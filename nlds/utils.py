from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import math
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = float(np.log(2.0 * math.pi))


def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.device, logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    r"""ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, shape[1] * shape[2])


def bernoulli_log_pdf(x, mu):
    r"""Log-likelihood of data given ~Bernoulli(mu)

    @param x: PyTorch.Tensor
              ground truth input
    @param mu: PyTorch.Tensor
               Bernoulli distribution parameters
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


def categorical_log_pdf(x, logits):
    r"""Log-likelihood of data given ~Bernoulli(mu)
    @param x: PyTorch.Tensor
              ground truth input
    @param logits: PyTorch.Tensor
               logits (pre-softmax)
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    return -F.cross_entropy(logits, x.long(), reduction='none')


def gaussian_log_pdf(x, mu, logvar):
    r"""Log-likelihood of data given ~N(mu, exp(logvar))
    log f(x) = log(1/sqrt(2*pi*var) * e^(-(x - mu)^2 / var))
             = -1/2 log(2*pi*var) - 1/2 * ((x-mu)/sigma)^2
             = -1/2 log(2pi) - 1/2log(var) - 1/2((x-mu)/sigma)^2
             = -1/2 log(2pi) - 1/2[((x-mu)/sigma)^2 + log var]
    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -LOG2PI * x.size(1) / 2. - \
        torch.sum(logvar + torch.pow(x - mu, 2) / (torch.exp(logvar) + 1e-7), dim=1) / 2.

    return log_pdf


def unit_gaussian_log_pdf(x):
    r"""Log-likelihood of data given ~N(0, 1)
    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -LOG2PI * x.size(1) / 2. - \
        torch.sum(torch.pow(x, 2), dim=1) / 2.

    return log_pdf


def log_mean_exp(x, dim=1):
    r"""log(1/k * sum(exp(x))): this normalizes x.

    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


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


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))
