"""
Generate as many Navier Stokes systems with random initial 
conditions and random boundary conditions.
"""

import os
import copy
import pickle
import numpy as np
from tqdm import tqdm
from src.navierstokes.flow import (
    LinearConvectionSystem,
    NonlinearConvectionSystem,
    DiffusionSystem,
    BurgersSystem,
    NavierStokesSystem,
    MomentumBoundaryCondition,
    PressureBoundaryCondition,
)

if 'ccncluster' in os.uname()[1]:
    DATA_DIR = '/mnt/fs5/wumike/navierstokes/data'
else:
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def generate_random_config(nt, nit, nx, ny, dt, rho, nu, c,  
                           constant_derivative=False):
    dx, dy = 2. / (nx - 1), 2. / (ny - 1)

    # randomly pick source 
    F = np.random.choice([0, 1], p=[0.8, 0.2])

    # randomly generate initial conditions
    u_ic = np.random.randn(nx, ny) * 0.01
    v_ic = np.random.randn(nx, ny) * 0.01
    p_ic = np.random.randn(nx, ny) * 0.01

    # create random boundary conditions
    u_bc_x0_lst, u_bc_xn_lst, u_bc_y0_lst, u_bc_yn_lst = [], [], [], []
    v_bc_x0_lst, v_bc_xn_lst, v_bc_y0_lst, v_bc_yn_lst = [], [], [], []
    p_bc_x0_lst, p_bc_xn_lst, p_bc_y0_lst, p_bc_yn_lst = [], [], [], []
    
    # Assumption for numerical stability: all the boundary elements in a row
    # or column boundary are identical. 

    u_x0_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    u_xn_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    u_x0_boundary_value = np.random.randn()
    u_xn_boundary_value = np.random.randn()
    u_x0_boundary_dict = {u_x0_boundary_type: u_x0_boundary_value}
    u_xn_boundary_dict = {u_xn_boundary_type: u_xn_boundary_value}

    v_x0_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    v_xn_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    v_x0_boundary_value = np.random.randn()
    v_xn_boundary_value = np.random.randn()
    v_x0_boundary_dict = {v_x0_boundary_type: v_x0_boundary_value}
    v_xn_boundary_dict = {v_xn_boundary_type: v_xn_boundary_value}

    p_x0_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    p_xn_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    p_x0_boundary_value = np.random.randn()
    p_xn_boundary_value = np.random.randn()
    p_x0_boundary_dict = {p_x0_boundary_type: p_x0_boundary_value}
    p_xn_boundary_dict = {p_xn_boundary_type: p_xn_boundary_value}

    for i in range(nx):
        u_bc_x0 = MomentumBoundaryCondition(i, 0, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **u_x0_boundary_dict)
        u_bc_xn = MomentumBoundaryCondition(i, ny - 1, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **u_xn_boundary_dict)

        v_bc_x0 = MomentumBoundaryCondition(i, 0, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **v_x0_boundary_dict)
        v_bc_xn = MomentumBoundaryCondition(i, ny - 1, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **v_xn_boundary_dict)

        p_bc_x0 = PressureBoundaryCondition(i, 0, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **p_x0_boundary_dict)
        p_bc_xn = PressureBoundaryCondition(i, ny - 1, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **p_xn_boundary_dict)

        # append in a very particular order
        u_bc_x0_lst.append(u_bc_x0)
        u_bc_xn_lst.append(u_bc_xn)
        v_bc_x0_lst.append(v_bc_x0)
        v_bc_xn_lst.append(v_bc_xn)
        p_bc_x0_lst.append(p_bc_x0)
        p_bc_xn_lst.append(p_bc_xn)

    u_y0_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    u_yn_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    u_y0_boundary_value = np.random.randn()
    u_yn_boundary_value = np.random.randn()
    u_y0_boundary_dict = {u_y0_boundary_type: u_y0_boundary_value}
    u_yn_boundary_dict = {u_yn_boundary_type: u_yn_boundary_value}

    v_y0_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    v_yn_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    v_y0_boundary_value = np.random.randn()
    v_yn_boundary_value = np.random.randn()
    v_y0_boundary_dict = {v_y0_boundary_type: v_y0_boundary_value}
    v_yn_boundary_dict = {v_yn_boundary_type: v_yn_boundary_value}

    p_y0_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    p_yn_boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
    p_y0_boundary_value = np.random.randn()
    p_yn_boundary_value = np.random.randn()
    p_y0_boundary_dict = {p_y0_boundary_type: p_y0_boundary_value}
    p_yn_boundary_dict = {p_yn_boundary_type: p_yn_boundary_value} 

    for j in range(ny):
        u_bc_y0 = MomentumBoundaryCondition(0, j, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **u_y0_boundary_dict)
        u_bc_yn = MomentumBoundaryCondition(nx - 1, j, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **u_yn_boundary_dict)

        v_bc_y0 = MomentumBoundaryCondition(0, j, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **v_y0_boundary_dict)
        v_bc_yn = MomentumBoundaryCondition(nx - 1, j, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **v_yn_boundary_dict)

        p_bc_y0 = PressureBoundaryCondition(0, j, 0, nx - 1, 0, ny - 1, 
                                            dx, dy, **p_y0_boundary_dict)
        p_bc_yn = PressureBoundaryCondition(nx - 1, j, 0, nx - 1, 0, ny - 1,
                                            dx, dy, **p_yn_boundary_dict)

        u_bc_y0_lst.append(u_bc_y0)
        u_bc_yn_lst.append(u_bc_yn)
        v_bc_y0_lst.append(v_bc_y0)
        v_bc_yn_lst.append(v_bc_yn)
        p_bc_y0_lst.append(p_bc_y0)
        p_bc_yn_lst.append(p_bc_yn)

    u_bc = u_bc_x0_lst + u_bc_xn_lst + u_bc_y0_lst + u_bc_yn_lst
    v_bc = v_bc_x0_lst + v_bc_xn_lst + v_bc_y0_lst + v_bc_yn_lst
    p_bc = p_bc_x0_lst + p_bc_xn_lst + p_bc_y0_lst + p_bc_yn_lst

    # this config should work with all systems
    config = {'u_ic': u_ic, 'v_ic': v_ic, 'p_ic': p_ic,
              'u_bc': u_bc, 'v_bc': v_bc, 'p_bc': p_bc,
              'nt': nt, 'nit': nit, 'nx': nx, 'ny': ny,
              'dt': dt, 'rho': rho, 'nu': nu, 'F': F, 
              'c': c, 'nu': nu, 'constant_derivative': constant_derivative}
    
    return config


def generate_system(system, config):
    if system == 'linear_convection' or system == 'linear':
        system = LinearConvectionSystem(config['u_ic'], config['u_bc'], 
                                        nt=config['nt'], nit=config['nit'], 
                                        nx=config['nx'], ny=config['ny'], 
                                        dt=config['dt'], c=config['c'],
                                        constant_derivative=config['constant_derivative'])
        u = system.simulate()
        return u, np.zeros_like(u), np.zeros_like(u)
    elif system == 'nonlinear_convection' or system == 'nonlinear':
        system = NonlinearConvectionSystem(config['u_ic'], config['v_ic'],
                                           config['u_bc'], config['v_bc'],
                                           nt=config['nt'], nit=config['nit'], 
                                           nx=config['nx'], ny=config['ny'], 
                                           dt=config['dt'],
                                           constant_derivative=config['constant_derivative'])
        u, v = system.simulate()
        return u, v, np.zeros_like(u)
    elif system == 'diffusion':
        system = DiffusionSystem(config['u_ic'], config['u_bc'], 
                                 nt=config['nt'], nit=config['nit'],
                                 nx=config['nx'], ny=config['ny'],
                                 dt=config['dt'], nu=config['nu'])
        u = system.simulate()
        return u, np.zeros_like(u), np.zeros_like(u)
    elif system == 'burgers':
        system = BurgersSystem(config['u_ic'], config['v_ic'], config['p_ic'],
                               config['u_bc'], config['v_bc'], config['p_bc'],
                               nt=config['nt'], nit=config['nit'],
                               nx=config['nx'], ny=config['ny'],
                               dt=config['dt'], nu=config['nu'])
        u, v = system.simulate()
        return u, v, np.zeros_like(u)
    elif system == 'navier_stokes':
        system = NavierStokesSystem(config['u_ic'], config['v_ic'], config['p_ic'], 
                                    config['u_bc'], config['v_bc'], config['p_bc'],
                                    nt=config['nt'], nit=config['nit'], 
                                    nx=config['nx'], ny=config['ny'], dt=config['dt'],
                                    rho=config['rho'], nu=config['nu'], F=config['F'])
        u, v, p = system.simulate()
        return u, v, p


def fine_to_coarse_config(fine_config, coarse_grid_dim):
    # NOTE: again keep this as general as possible!
    # make a coarse config from the fine config
    coarse_config = copy.deepcopy(fine_config)
    fine_grid_dim = fine_config['nx']
    coarse_config['nx'] = coarse_grid_dim; coarse_config['ny'] = coarse_grid_dim
    coarse_config['u_ic'] = np.zeros((coarse_config['nx'], coarse_config['ny']))
    coarse_config['v_ic'] = np.zeros((coarse_config['nx'], coarse_config['ny']))
    coarse_config['p_ic'] = np.zeros((coarse_config['nx'], coarse_config['ny']))
    const = fine_config['nx'] // coarse_config['nx']
    # subsample the boundary conditions
    interval = np.arange(0, fine_config['nx'], const)
    
    u_bc = copy.deepcopy(coarse_config['u_bc'])  # these are split equally into x0, xn, y0, yn
    v_bc = copy.deepcopy(coarse_config['v_bc'])
    p_bc = copy.deepcopy(coarse_config['p_bc'])
    length = len(u_bc)
    
    # split into 4 sets for the 4 sides of a square
    u_bc_x0, u_bc_xn, u_bc_y0, u_bc_yn = (u_bc[:length//4], u_bc[length//4:length//2], 
                                          u_bc[length//2:3*length//4], u_bc[3*length//4:])
    v_bc_x0, v_bc_xn, v_bc_y0, v_bc_yn = (v_bc[:length//4], v_bc[length//4:length//2], 
                                          v_bc[length//2:3*length//4], v_bc[3*length//4:])
    p_bc_x0, p_bc_xn, p_bc_y0, p_bc_yn = (p_bc[:length//4], p_bc[length//4:length//2], 
                                          p_bc[length//2:3*length//4], p_bc[3*length//4:])

    # sample each one
    _u_bc_x0, _u_bc_xn, _u_bc_y0, _u_bc_yn = [], [], [], []
    _v_bc_x0, _v_bc_xn, _v_bc_y0, _v_bc_yn = [], [], [], []
    _p_bc_x0, _p_bc_xn, _p_bc_y0, _p_bc_yn = [], [], [], []

    for i in range(fine_config['nx']):
        if i % const == 0:
            u_bc_x0_i = copy.deepcopy(u_bc_x0[i])
            u_bc_x0_i.i = i // const
            u_bc_x0_i.j = u_bc_x0_i.j // const
            u_bc_x0_i.max_i = u_bc_x0_i.max_i // const
            u_bc_x0_i.max_j = u_bc_x0_i.max_j // const
            _u_bc_x0.append(u_bc_x0_i)
            
            u_bc_xn_i = copy.deepcopy(u_bc_xn[i])
            u_bc_xn_i.i = i // const
            u_bc_xn_i.j = u_bc_xn_i.j // const
            u_bc_xn_i.max_i = u_bc_xn_i.max_i // const
            u_bc_xn_i.max_j = u_bc_xn_i.max_j // const
            _u_bc_xn.append(u_bc_xn_i)
            
            u_bc_y0_i = copy.deepcopy(u_bc_y0[i])
            u_bc_y0_i.i = u_bc_y0_i.i // const
            u_bc_y0_i.j = i // const
            u_bc_y0_i.max_i = u_bc_x0_i.max_i // const
            u_bc_y0_i.max_j = u_bc_y0_i.max_j // const
            _u_bc_y0.append(u_bc_y0_i)
            
            u_bc_yn_i = copy.deepcopy(u_bc_yn[i])
            u_bc_yn_i.i = u_bc_yn_i.i // const
            u_bc_yn_i.j = i // const
            u_bc_yn_i.max_i = u_bc_yn_i.max_i // const
            u_bc_yn_i.max_j = u_bc_yn_i.max_j // const
            _u_bc_yn.append(u_bc_yn_i)

            # do same for v
            v_bc_x0_i = copy.deepcopy(v_bc_x0[i])
            v_bc_x0_i.i = i // const
            v_bc_x0_i.j = v_bc_x0_i.j // const
            v_bc_x0_i.max_i = v_bc_x0_i.max_i // const
            v_bc_x0_i.max_j = v_bc_x0_i.max_j // const
            _v_bc_x0.append(v_bc_x0_i)

            v_bc_xn_i = copy.deepcopy(v_bc_xn[i])
            v_bc_xn_i.i = i // const
            v_bc_xn_i.j = v_bc_xn_i.j // const
            v_bc_xn_i.max_i = v_bc_xn_i.max_i // const
            v_bc_xn_i.max_j = v_bc_xn_i.max_j // const
            _v_bc_xn.append(v_bc_xn_i)

            v_bc_y0_i = copy.deepcopy(v_bc_y0[i])
            v_bc_y0_i.i = v_bc_y0_i.i // const
            v_bc_y0_i.j = i // const
            v_bc_y0_i.max_i = v_bc_y0_i.max_i // const
            v_bc_y0_i.max_j = v_bc_y0_i.max_j // const
            _v_bc_y0.append(v_bc_y0_i)

            v_bc_yn_i = copy.deepcopy(v_bc_yn[i])
            v_bc_yn_i.i = v_bc_yn_i.i // const
            v_bc_yn_i.j = i // const
            v_bc_yn_i.max_i = v_bc_yn_i.max_i // const
            v_bc_yn_i.max_j = v_bc_yn_i.max_j // const
            _v_bc_yn.append(v_bc_yn_i)

            # finally, do same for p
            p_bc_x0_i = copy.deepcopy(p_bc_x0[i])
            p_bc_x0_i.i = i // const
            p_bc_x0_i.j = p_bc_x0_i.j // const
            p_bc_x0_i.max_i = p_bc_x0_i.max_i // const
            p_bc_x0_i.max_j = p_bc_x0_i.max_j // const
            _p_bc_x0.append(p_bc_x0_i)

            p_bc_xn_i = copy.deepcopy(p_bc_xn[i])
            p_bc_xn_i.i = i // const
            p_bc_xn_i.j = p_bc_xn_i.j // const
            p_bc_xn_i.max_i = p_bc_xn_i.max_i // const
            p_bc_xn_i.max_j = p_bc_xn_i.max_j // const
            _p_bc_xn.append(p_bc_xn_i)

            p_bc_y0_i = copy.deepcopy(p_bc_y0[i])
            p_bc_y0_i.i = p_bc_y0_i.i // const
            p_bc_y0_i.j = i // const
            p_bc_y0_i.max_i = p_bc_y0_i.max_i // const
            p_bc_y0_i.max_j = p_bc_y0_i.max_j // const
            _p_bc_y0.append(p_bc_y0_i)

            p_bc_yn_i = copy.deepcopy(p_bc_yn[i])
            p_bc_yn_i.i = p_bc_yn_i.i // const
            p_bc_yn_i.j = i // const
            p_bc_yn_i.max_i = p_bc_yn_i.max_i // const
            p_bc_yn_i.max_j = p_bc_yn_i.max_j // const
            _p_bc_yn.append(p_bc_yn_i)

    _u_bc = _u_bc_x0 + _u_bc_xn + _u_bc_y0 + _u_bc_yn
    _v_bc = _v_bc_x0 + _v_bc_xn + _v_bc_y0 + _v_bc_yn
    _p_bc = _p_bc_x0 + _p_bc_xn + _p_bc_y0 + _p_bc_yn

    # replace ones in coarse config with these
    coarse_config['u_bc'] = _u_bc
    coarse_config['v_bc'] = _v_bc
    coarse_config['p_bc'] = _p_bc

    return coarse_config


def check_system(u, v, p):
    pass_check = True
    if u is not None:
        if np.sum(np.isnan(u)) > 0:
            pass_check = False
    if v is not None:
        if np.sum(np.isnan(v)) > 0:
            pass_check = False
    if p is not None:
        if np.sum(np.isnan(p)) > 0:
            pass_check = False
    
    return pass_check


def save_system(save_path, system):
    np.savez(save_path, **system)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, cm
    from mpl_toolkits.mplot3d import Axes3D

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='navierstokes',
                        help='linear|nonlinear|linear_convection|nonlinear_convection|diffusion|burgers|navier_stokes')
    parser.add_argument('--num', type=int, default=1000,
                        help='number of systems to generate (default: 1000)')
    parser.add_argument('--fine-grid-dim', type=int, default=50,
                        help='granularity of fine grid [default: 50]')
    parser.add_argument('--coarse-grid-dim', type=int, default=10,
                        help='granularity of the coarse grid [dedafult: 10]')
    parser.add_argument('--block-size', type=int, default=100,
                        help='store N systems together (default: 100)')
    args = parser.parse_args()

    np.random.seed(1337)

    # these are fixed hyperparameters
    nt, nit, nx, ny = 200, 50, args.fine_grid_dim, args.fine_grid_dim 
    dt, rho, nu, c = 0.001, 1, 0.1, 1

    data_dir = os.path.join(DATA_DIR, args.system)
    image_dir = os.path.join(data_dir, 'images')
    numpy_dir = os.path.join(data_dir, 'numpy')
    fine_dir = os.path.join(numpy_dir, 'fine')
    coarse_dir = os.path.join(numpy_dir, 'coarse')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(fine_dir, exist_ok=True)
    os.makedirs(coarse_dir, exist_ok=True)

    constant_derivative = (True if args.system in ['linear', 'nonlinear'] else False)

    count = 0
    block_count = 0
    fine_systems_u, fine_systems_v, fine_systems_p = [], [], []
    coarse_systems_u, coarse_systems_v, coarse_systems_p = [], [], []

    while count < args.num:
        fine_config = generate_random_config(nt, nit, nx, ny, dt, rho, nu, c,  
                                             constant_derivative=constant_derivative)
        print('Generating **fine** {} system: ({}/{})'.format(args.system, count + 1, args.num))
        fine_u, fine_v, fine_p = generate_system(args.system, fine_config)  # make fine system!
        
        # lots of logic to preserve random choices but subsample
        coarse_config = fine_to_coarse_config(fine_config, args.coarse_grid_dim)
        print('Generating **coarse** {} system: ({}/{})'.format(args.system, count + 1, args.num))
        coarse_u, coarse_v, coarse_p = generate_system(args.system, coarse_config)

        # randomly initializing boundary conditions sometimes gets us into
        # trouble, so ignore when that happens.
        if not check_system(fine_u, fine_v, fine_p):
            continue

        if not check_system(coarse_u, coarse_v, coarse_p):
            continue

        fine_systems_u.append(fine_u)
        fine_systems_v.append(fine_v)
        fine_systems_p.append(fine_p)
        coarse_systems_u.append(coarse_u)
        coarse_systems_v.append(coarse_v)
        coarse_systems_p.append(coarse_p)
        
        count += 1

        if len(fine_systems) > args.block_size:
            fine_systems_u = np.stack(fine_systems_u)
            fine_systems_v = np.stack(fine_systems_v)
            fine_systems_p = np.stack(fine_systems_p)
            coarse_systems_u = np.stack(coarse_systems_u)
            coarse_systems_v = np.stack(coarse_systems_v)
            coarse_systems_p = np.stack(coarse_systems_p)

            save_system(os.path.join(fine_dir, 'system_{}.npz'.format(block_count)),
                u=fine_systems_u, v=fine_systems_v, p=fine_systems_p)
            save_system(os.path.join(coarse_dir, 'system_{}.npz'.format(block_count)),
                u=coarse_systems_u, v=coarse_systems_v, p=coarse_systems_p)

            fine_systems_u, fine_systems_v, fine_systems_p = [], [], []
            coarse_systems_u, coarse_systems_v, coarse_systems_p = [], [], []

            block_count += 1
