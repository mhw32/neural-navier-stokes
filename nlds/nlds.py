r"""Switching Recurrent Nonlinear Dynamical System."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn

from dmm import DMM


class NLDS(nn.Module):
    r"""Switching Recurrent Nonlinear Dynamical System.
    
    "Categorical" distribution over several nonlinear dynamical systems,
    each of which is modeled by a DMM. 

    Natural generalization to https://arxiv.org/pdf/1610.08466.pdf.

    We want this to be trainable end-to-end by gradient descent. So, we 
    will rely on a Gumbel softmax parameterization of a categorical.
    """
    pass
