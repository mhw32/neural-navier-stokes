"""Neural Switching-state Linear Dynamical Model"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.spiral.dataset import generate_spiral2d
from src.spiral.utils import AverageMeter, log_normal_pdf, normal_kl
from src.spiral.ldm import reverse_sequences_torch
