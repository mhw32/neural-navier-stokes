import os
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D

from src.navierstokes.generate import DATA_DIR
from src.navierstokes.utils import load_systems
from src.navierstokes.baseline import coarsen_fine_systems


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='navier_stokes',
                        help='linear|nonlinear|linear_convection|nonlinear_convection|diffusion|burgers|navier_stokes')
    args = parser.parse_args()

    # fine
    data_dir = os.path.join(DATA_DIR, args.system)
    image_dir = os.path.join(data_dir, 'images')
    u_fine, v_fine, p_fine = load_systems(data_dir, fine=True)

    # coarse
    u_coarse, v_coarse, p_coarse = load_systems(data_dir, fine=False)

    # coarsened
    N = u_fine.shape[0]
    nx, ny = u_fine.shape[2], u_fine.shape[3]
    x_fine = np.linspace(0, 2, nx)  # slightly hardcoded
    y_fine = np.linspace(0, 2, ny)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    u_coarsened, v_coarsened, p_coarsened = coarsen_fine_systems(
        X_fine, Y_fine, u_fine, v_fine, p_fine)

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    print('Plotting functions')
    n_plot = 10
    for i in tqdm(range(n_plot)):
        u, v, p = u_fine[i], v_fine[i], p_fine[i]
        fig = plt.figure(figsize=(11, 7), dpi=100)
        plt.contourf(X, Y, p[-1], alpha=0.5, cmap=cm.viridis)
        plt.colorbar()
        plt.contour(X, Y, p[-1], cmap=cm.viridis)
        plt.streamplot(X, Y, u[-1], v[-1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(image_dir, 'fine_system_streamplot_{}.pdf'.format(i)))
        plt.close()

    x = np.linspace(0, 2, nx // 5)
    y = np.linspace(0, 2, ny // 5)
    X, Y = np.meshgrid(x, y)

    for i in tqdm(range(n_plot)):
        u, v, p = u_coarse[i], v_coarse[i], p_coarse[i]
        fig = plt.figure(figsize=(11, 7), dpi=100)
        plt.contourf(X, Y, p[-1], alpha=0.5, cmap=cm.viridis)
        plt.colorbar()
        plt.contour(X, Y, p[-1], cmap=cm.viridis)
        plt.streamplot(X, Y, u[-1], v[-1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(image_dir, 'coarse_system_streamplot_{}.pdf'.format(i)))
        plt.close()

    for i in tqdm(range(n_plot)):
        u, v, p = u_coarsened[i], v_coarsened[i], p_coarsened[i]
        fig = plt.figure(figsize=(11, 7), dpi=100)
        plt.contourf(X, Y, p[-1], alpha=0.5, cmap=cm.viridis)
        plt.colorbar()
        plt.contour(X, Y, p[-1], cmap=cm.viridis)
        plt.streamplot(X, Y, u[-1], v[-1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(image_dir, 'coarsened_system_streamplot_{}.pdf'.format(i)))
        plt.close()
