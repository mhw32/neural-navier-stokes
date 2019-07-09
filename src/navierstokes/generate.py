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
    u_bc_x0_lst, u_bc_xn_lst, u_bc_y0_lst, u_bc_yn_lst = [], [], [], []
    v_bc_x0_lst, v_bc_xn_lst, v_bc_y0_lst, v_bc_yn_lst = [], [], [], []
    p_bc_x0_lst, p_bc_xn_lst, p_bc_y0_lst, p_bc_yn_lst = [], [], [], []
    
    for i in range(nx):
        boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
        boundary_value = np.random.randn()
        boundary_dict = {boundary_type: boundary_value}

        u_bc_x0 = MomentumBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc_xn = MomentumBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)

        v_bc_x0 = MomentumBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc_xn = MomentumBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)

        p_bc_x0 = PressureBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc_xn = PressureBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)

        # append in a very particular order
        u_bc_x0_lst.append(u_bc_x0)
        u_bc_xn_lst.append(u_bc_xn)
        v_bc_x0_lst.append(v_bc_x0)
        v_bc_xn_lst.append(v_bc_xn)
        p_bc_x0_lst.append(p_bc_x0)
        p_bc_xn_lst.append(p_bc_xn)

    for j in range(ny):
        u_bc_y0 = MomentumBoundaryCondition(0, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc_yn = MomentumBoundaryCondition(2, j, 0, 2, 0, 2, dx, dy, **boundary_dict)

        v_bc_y0 = MomentumBoundaryCondition(0, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc_yn = MomentumBoundaryCondition(2, j, 0, 2, 0, 2, dx, dy, **boundary_dict)

        p_bc_y0 = PressureBoundaryCondition(0, j, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc_yn = PressureBoundaryCondition(2, j, 0, 2, 0, 2, dx, dy, **boundary_dict)

        u_bc_y0_lst.append(u_bc_y0)
        u_bc_yn_lst.append(u_bc_yn)
        v_bc_y0_lst.append(v_bc_y0)
        v_bc_yn_lst.append(v_bc_yn)
        p_bc_y0_lst.append(p_bc_y0)
        p_bc_yn_lst.append(p_bc_yn)

    u_bc = u_bc_x0_lst + u_bc_xn_lst + u_bc_y0_lst + u_bc_yn_lst
    v_bc = v_bc_x0_lst + v_bc_xn_lst + v_bc_y0_lst + v_bc_yn_lst
    p_bc = p_bc_x0_lst + p_bc_xn_lst + p_bc_y0_lst + p_bc_yn_lst

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


def fine_to_coarse_config(fine_config):
    # make a coarse config from the fine config
    coarse_config = copy.deepcopy(fine_config)
    coarse_config['nx'] = 10; coarse_config['ny'] = 10  # hardcoded!
    coarse_config['u_ic'] = np.zeros((coarse_config['nx'], coarse_config['ny']))
    coarse_config['v_ic'] = np.zeros((coarse_config['nx'], coarse_config['ny']))
    coarse_config['p_ic'] = np.zeros((coarse_config['nx'], coarse_config['ny']))
    const = fine_config['nx'] // coarse_config['nx']
    # subsample the boundary conditions
    interval = np.arange(0, fine_config['nx'], const)
    
    u_bc = copy.deepcopy(coarse_config['u_bc'])  # these are split equally into x0, xn, y0, yn
    v_bc = copy.deepcopy(coarse_config['v_bc'])
    p_bc = copy.deepcopy(coarse_config['p_bc'])

    u_bc_x0, u_bc_xn, u_bc_y0, u_bc_yn = u_bc[:50], u_bc[50:100], u_bc[100:150], u_bc[150:]
    v_bc_x0, v_bc_xn, v_bc_y0, v_bc_yn = v_bc[:50], v_bc[50:100], v_bc[100:150], v_bc[150:]
    p_bc_x0, p_bc_xn, p_bc_y0, p_bc_yn = p_bc[:50], p_bc[50:100], p_bc[100:150], p_bc[150:]

    # sample each one
    _u_bc_x0, _u_bc_xn, _u_bc_y0, _u_bc_yn = [], [], [], []
    _v_bc_x0, _v_bc_xn, _v_bc_y0, _v_bc_yn = [], [], [], []
    _p_bc_x0, _p_bc_xn, _p_bc_y0, _p_bc_yn = [], [], [], []

    for i in range(fine_config['nx']):
        if i % 5 == 0:
            _u_bc_x0.append(u_bc_x0[i])
            _u_bc_xn.append(u_bc_xn[i])
            _u_bc_y0.append(u_bc_y0[i])
            _u_bc_yn.append(u_bc_yn[i])
            _v_bc_x0.append(v_bc_x0[i])
            _v_bc_xn.append(v_bc_xn[i])
            _v_bc_y0.append(v_bc_y0[i])
            _v_bc_yn.append(v_bc_yn[i])
            _p_bc_x0.append(p_bc_x0[i])
            _p_bc_xn.append(p_bc_xn[i])
            _p_bc_y0.append(p_bc_y0[i])
            _p_bc_yn.append(p_bc_yn[i])

    _u_bc = _u_bc_x0 + _u_bc_xn + _u_bc_y0 + _u_bc_yn
    _v_bc = _v_bc_x0 + _v_bc_xn + _v_bc_y0 + _v_bc_yn
    _p_bc = _p_bc_x0 + _p_bc_xn + _p_bc_y0 + _p_bc_yn

    # replace ones in coarse config with these
    coarse_config['u_bc'] = _u_bc
    coarse_config['v_bc'] = _v_bc
    coarse_config['p_bc'] = _p_bc

    return coarse_config


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
        fine_config = generate_random_config(nt, nit, nx, ny, dt, rho, nu)
        print('Generating **fine** navier-stokes system: ({}/{})'.format(count + 1, args.num))
        fine_system = generate_system(fine_config)  # make fine system!
        
        # lots of logic to preserve random choices but subsample
        coarse_config = fine_to_coarse_config(fine_config)
        print('Generating **coarse** navier-stokes system: ({}/{})'.format(count + 1, args.num))
        coarse_system = generate_system(coarse_config)

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
