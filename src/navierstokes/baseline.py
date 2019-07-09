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
from src.navierstokes.utils import dynamics_prediction_error_numpy
from src.navierstokes.utils import spatial_coarsen, AverageMeter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


if __name__ == "__main__":
    np.random.seed(1337)

    with open(os.path.join(DATA_DIR, '1000_fine_systems.pickle'), 'rb') as fp:
        fine_systems = pickle.load(fp)

    with open(os.path.join(DATA_DIR, '1000_coarse_systems.pickle'), 'rb') as fp:
        coarse_systems = pickle.load(fp)

    fine_config = fine_systems[0]['config']
    coarse_config = coarse_systems[0]['config']

    fine_x = np.linspace(0, 2, fine_config['nx'])
    fine_y = np.linspace(0, 2, fine_config['ny'])
    fine_X, fine_Y = np.meshgrid(fine_x, fine_y)

    agg_x = fine_config['nx'] // coarse_config['nx']
    agg_y = fine_config['ny'] // coarse_config['ny']

    # coarsen the fine systems
    print('Coarsening "fine" systems:')
    coarsened_systems = []
    for i in tqdm(range(len(fine_systems))):
        u_seq, v_seq, p_seq = (fine_systems[i]['u'],
                               fine_systems[i]['v'], 
                               fine_systems[i]['p'])
        _, _, u_seq, v_seq, p_seq = spatial_coarsen(
            fine_X, fine_Y, u_seq, v_seq, p_seq,
            agg_x=agg_x, agg_y=agg_y)

        coarsened_system_i = copy.deepcopy(fine_systems[i])
        coarsened_system_i['config']['nx'] = coarse_config['nx']
        coarsened_system_i['config']['ny'] = coarse_config['ny']
        coarsened_system_i['u'] = u_seq
        coarsened_system_i['v'] = v_seq
        coarsened_system_i['p'] = p_seq
        coarsened_systems.append(coarsened_system_i)

    print('Computing error for coarse systems:')
    u_errors, v_errors, p_errors = [], [], []
    for i in tqdm(range(len(coarse_systems))):
        u_seq, v_seq, p_seq = (coarse_systems[i]['u'],
                               coarse_systems[i]['v'], 
                               coarse_systems[i]['p'])
        u_hat_seq, v_hat_seq, p_hat_seq = (
            coarsened_systems[i]['u'], coarsened_systems[i]['v'], 
            coarsened_systems[i]['p'])

        u_error_i, v_error_i, p_error_i = dynamics_prediction_error_numpy(
            u_seq, v_seq, p_seq, u_hat_seq, v_hat_seq, p_hat_seq)
        u_errors.append(u_error_i)
        v_errors.append(v_error_i)
        p_errors.append(p_error_i)
    # this will be (1000, T)
    u_errors = np.stack(u_errors, axis=0)
    v_errors = np.stack(v_errors, axis=0)
    p_errors = np.stack(p_errors, axis=0)

    u_error_mean = np.mean(u_errors, axis=0)
    u_error_stdev = np.std(u_errors, axis=0)
    v_error_mean = np.mean(v_errors, axis=0)
    v_error_stdev = np.std(v_errors, axis=0)
    p_error_mean = np.mean(p_errors, axis=0)
    p_error_stdev = np.std(p_errors, axis=0)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, 'baseline_error.npz'), 
             u_mean=u_error_mean, u_std=u_error_stdev,
             v_mean=v_error_mean, v_std=v_error_stdev,
             p_mean=p_error_mean, p_std=p_error_stdev)
