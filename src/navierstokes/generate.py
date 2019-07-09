"""
Generate as many Navier Stokes systems with random initial 
conditions and random boundary conditions.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from src.navierstokes.flow import (
    NavierStokesSystem,
    MomentumBoundaryCondition,
    PressureBoundaryCondition,
)

if 'ccncluster' in os.uname()[1]:
    DATA_DIR = '/mnt/fs5/wumike/navierstokes/data'
    IMAGE_DIR = '/mnt/fs5/wumike/navierstokes/data/images'
else:
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'data/images')


def generate_random_config(nt, nit, nx, ny, dt, rho, nu):
    dx, dy = 2. / (nx - 1), 2. / (ny - 1)

    # randomly pick source 
    F = np.random.choice([0, 1], p=[0.8, 0.2])

    # for now, we will use the same initial 
    # conditions (to highlight differences in boundary)
    u_ic = np.zeros((nx, ny))
    v_ic = np.zeros((nx, ny))
    p_ic = np.zeros((nx, ny))

    # create random boundary conditions
    u_bc, v_bc, p_bc = [], [], []
    for i in range(nx):
        boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
        boundary_value = np.random.randn()
        boundary_dict = {boundary_type: boundary_value}

        u_bc_x0 = MomentumBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc_xn = MomentumBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc.extend([u_bc_x0, u_bc_xn])

        v_bc_x0 = MomentumBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc_xn = MomentumBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc.extend([v_bc_x0, v_bc_xn])

        p_bc_x0 = PressureBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc_xn = PressureBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc.extend([p_bc_x0, p_bc_xn])

    for j in range(ny):
        u_bc_y0 = MomentumBoundaryCondition(0, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc_yn = MomentumBoundaryCondition(2, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc.extend([u_bc_y0, u_bc_yn])

        v_bc_y0 = MomentumBoundaryCondition(0, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc_yn = MomentumBoundaryCondition(2, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc.extend([v_bc_y0, v_bc_yn])

        p_bc_y0 = PressureBoundaryCondition(0, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc_yn = PressureBoundaryCondition(2, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc.extend([p_bc_y0, p_bc_yn])

    config = {'u_ic': u_ic, 'v_ic': v_ic, 'p_ic': p_ic,
              'u_bc': u_bc, 'v_bc': v_bc, 'p_bc': p_bc,
              'nt': nt, 'nit': nit, 'nx': nx, 'ny': ny,
              'dt': dt, 'rho': rho, 'nu': nu, 'F': F}
    
    return config


def generate_system(config):
    system = NavierStokesSystem(config['u_ic'], config['v_ic'], config['p_ic'], 
                                config['u_bc'], config['v_bc'], config['p_bc'],
                                nt=config['nt'], nit=config['nit'], 
                                nx=config['nx'], ny=config['ny'], dt=config['dt'],
                                rho=config['rho'], nu=config['nu'], F=config['F'])
    u, v, p = system.simulate()
    return {'u': u, 'v': v, 'p': p, 'config': config}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, cm
    from mpl_toolkits.mplot3d import Axes3D

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=100,
                        help='number of systems to generate (default: 100)')
    args = parser.parse_args()

    # these are fixed hyperparameters
    nt, nit, nx, ny = 200, 50, 50, 50
    dt, rho, nu = 0.001, 1, 0.1

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    count = 0
    fine_systems, coarse_systems = [], []
    while count < args.num:
        config = generate_random_config(nt, nit, nx, ny, dt, rho, nu)
        print('Generating **fine** navier-stokes system: ({}/{})'.format(count + 1, args.num))
        fine_system = generate_system(config)  # make fine system!
        config['nx'] = 10; config['ny'] = 10   # make coarse system!
        print('Generating **coarse** navier-stokes system: ({}/{})'.format(count + 1, args.num))
        coarse_system = generate_system(config)

        # randomly initializing boundary conditions sometimes gets us into
        # trouble, so ignore when that happens.
        if (np.sum(np.isnan(fine_system['u'])) > 0 or 
            np.sum(np.isnan(fine_system['v'])) > 0 or 
            np.sum(np.isnan(fine_system['p'])) > 0):
            continue

        if (np.sum(np.isnan(coarse_system['u'])) > 0 or 
            np.sum(np.isnan(coarse_system['v'])) > 0 or 
            np.sum(np.isnan(coarse_system['p'])) > 0):
            continue

        fine_systems.append(fine_system)
        coarse_systems.append(coarse_system)
        count += 1

    with open(os.path.join(DATA_DIR, '{}_fine_systems.pickle'.format(args.num)), 'wb') as fp:
        pickle.dump(fine_systems, fp)

    with open(os.path.join(DATA_DIR, '{}_coarse_systems.pickle'.format(args.num)), 'wb') as fp:
        pickle.dump(coarse_systems, fp)

    # for each system, save image so we can get a sense of
    # how different these things are

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    print('Plotting functions')
    for i in tqdm(range(100)):
        u, v, p = fine_systems[i]['u'], fine_systems[i]['v'], fine_systems[i]['p']
        fig = plt.figure(figsize=(11, 7), dpi=100)
        plt.contourf(X, Y, p[-1], alpha=0.5, cmap=cm.viridis)
        plt.colorbar()
        plt.contour(X, Y, p[-1], cmap=cm.viridis)
        plt.streamplot(X, Y, u[-1], v[-1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(IMAGE_DIR, 'streamplot_system_{}.pdf'.format(i)))
        plt.close()
