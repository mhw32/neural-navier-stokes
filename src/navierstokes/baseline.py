"""
As a pretty strong baseline, we will use the "coarse" 
meshgrid numerical approximation to Navier Stokes as 
a predictor for the dynamics of the "coarsened-fine" 
meshgrid. This intuitively represents spending extra
compute (no generalization).
"""

import os
import copy
import pickle
import numpy as np
from tqdm import tqdm

from src.navierstokes.generate import DATA_DIR
from src.navierstokes.utils import (dynamics_prediction_error_numpy,
                                    spatial_coarsen, AverageMeter, load_systems)


def coarsen_fine_systems(X_fine, Y_fine, u_fine, v_fine, p_fine):
    # coarsen the fine systems
    print('Coarsening "fine" systems:')
    u_coarsened, v_coarsened, p_coarsened = [], [], []
    for i in tqdm(range(len(u_fine))):
        u_seq, v_seq, p_seq = u_fine[i], v_fine[i], p_fine[i]
        _, _, u_seq, v_seq, p_seq = spatial_coarsen(
            X_fine, Y_fine, u_seq, v_seq, p_seq,
            agg_x=5, agg_y=5)  # HARDCODE for now

        u_coarsened.append(u_seq.copy())
        v_coarsened.append(v_seq.copy())
        p_coarsened.append(p_seq.copy())
    
    u_coarsened = np.stack(u_coarsened)
    v_coarsened = np.stack(v_coarsened)
    p_coarsened = np.stack(p_coarsened)

    return u_coarsened, v_coarsened, p_coarsened


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='navier_stokes',
                                    help='linear|nonlinear|linear_convection|nonlinear_convection|diffusion|burgers|navier_stokes')
    args = parser.parse_args()

    np.random.seed(1337)

    data_dir = os.path.join(DATA_DIR, args.system)
    u_fine, v_fine, p_fine = load_systems(data_dir, fine=True)
    u_coarse, v_coarse, p_coarse = load_systems(data_dir, fine=False)
    
    N = u_fine.shape[0]
    N_train = int(0.8 * N)
    N_val = int(0.1 * N)

    # just grab the ``test set"
    u_fine = u_fine[N_train + N_val:, ...]
    v_fine = v_fine[N_train + N_val:, ...]
    p_fine = p_fine[N_train + N_val:, ...]
    
    u_coarse = u_coarse[N_train + N_val:, ...]
    v_coarse = v_coarse[N_train + N_val:, ...]
    p_coarse = p_coarse[N_train + N_val:, ...]

    N = len(u_fine)
    nx, ny = u_fine.shape[2], u_fine.shape[3]
    x_fine = np.linspace(0, 2, nx)  # slightly hardcoded
    y_fine = np.linspace(0, 2, ny)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    u_coarsened, v_coarsened, p_coarsened = coarsen_fine_systems(
        X_fine, Y_fine, u_fine, v_fine, p_fine)

    print('Computing error for coarse systems:')
    u_errors, v_errors, p_errors = [], [], []
    for i in tqdm(range(N)):
        u_seq, v_seq, p_seq = u_coarse[i], v_coarse[i], p_coarse[i]
        u_hat_seq, v_hat_seq, p_hat_seq = u_coarsened[i], v_coarsened[i], p_coarsened[i]
        u_error_i, v_error_i, p_error_i = dynamics_prediction_error_numpy(
            u_seq, v_seq, p_seq, u_hat_seq, v_hat_seq, p_hat_seq)
        u_errors.append(u_error_i)
        v_errors.append(v_error_i)
        p_errors.append(p_error_i)

    u_errors = np.stack(u_errors)
    v_errors = np.stack(v_errors)
    p_errors = np.stack(p_errors)

    u_err_mean = np.mean(u_errors, axis=0)
    v_err_mean = np.mean(v_errors, axis=0)
    p_err_mean = np.mean(p_errors, axis=0)

    u_err_std = np.std(u_errors, axis=0)
    v_err_std = np.std(v_errors, axis=0)
    p_err_std = np.std(p_errors, axis=0)

    np.savez(os.path.join(data_dir, 'baseline_error.npz'), 
             u_error_mean=u_err_mean, v_error_mean=v_err_mean,
             p_error_mean=p_err_mean, u_error_std=u_err_std,
             v_error_std=v_err_std, p_error_std=p_err_std)
