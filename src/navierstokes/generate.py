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
DATA_DIR = '/mnt/fs5/wumike/navierstokes/data'


def generate_random_system(nt, nit, nx, ny, dt, rho, nu):
    dx, dy = 2. / (nx - 1), 2. / (ny - 1)

    # randomly pick source 
    F = np.random.choice([0, 1], p=[0.8, 0.2])

    # create random initial conditions
    u_ic = np.random.randn(nx, ny)
    v_ic = np.random.randn(nx, ny)
    p_ic = np.random.uniform(0, 1, size=nx*ny).reshape(nx, ny)

    # create random boundary conditions
    u_bc, v_bc, p_bc = [], [], []
    for i in np.linspace(0, 2, nx):
        boundary_type = np.random.choice(['dirichlet', 'neumann'], p=[0.5, 0.5])
        boundary_value = np.random.randn()
        boundary_dict = {boundary_type: boundary_value}

        u_bc_y0 = MomentumBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc_yn = MomentumBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc_x0 = MomentumBoundaryCondition(0, i, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc_xn = MomentumBoundaryCondition(2, i, 0, 2, 0, 2, dx, dy, **boundary_dict)
        u_bc.extend([u_bc_y0, u_bc_yn, u_bc_x0, u_bc_xn])

        v_bc_y0 = MomentumBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc_yn = MomentumBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc_x0 = MomentumBoundaryCondition(0, i, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc_xn = MomentumBoundaryCondition(2, i, 0, 2, 0, 2, dx, dy, **boundary_dict)
        v_bc.extend([v_bc_y0, v_bc_yn, v_bc_x0, v_bc_xn])

        p_bc_y0 = PressureBoundaryCondition(i, 0, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc_yn = PressureBoundaryCondition(i, 2, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc_x0 = PressureBoundaryCondition(0, i, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc_xn = PressureBoundaryCondition(2, i, 0, 2, 0, 2, dx, dy, **boundary_dict)
        p_bc.extend([p_bc_y0, p_bc_yn, p_bc_x0, p_bc_xn])

    config = {'u_ic': u_ic, 'v_ic': v_ic, 'p_ic': p_ic,
              'u_bc': u_bc, 'v_bc': v_bc, 'p_bc': p_bc,
              'nt': nt, 'nit': nit, 'nx': nx, 'ny': ny,
              'dt': dt, 'rho': rho, 'nu': nu, 'F': F}
    system = NavierStokesSystem(u_ic, v_ic, p_ic, u_bc, v_bc, p_bc,
                                nt=nt, nit=nit, nx=nx, ny=ny, dt=dt,
                                rho=rho, nu=nu, F=F)
    u, v, p = system.simulate()
    return {'u': u, 'v': v, 'p': p, 'config': config}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=100,
                        help='number of systems to generate (default: 100)')
    args = parser.parse_args()

    # these are fixed hyperparameters
    nt, nit, nx, ny = 200, 50, 50, 50
    dt, rho, nu = 0.001, 1, 0.1

    systems = []
    for i in tqdm(range(args.num)):
        system = generate_random_system(nt, nit, nx, ny, dt, rho, nu)
        systems.append(system)

    with open(os.path.join(DATA_DIR, '{}_systems.pickle'), 'wb') as fp:
        pickle.dump(systems, fp)
