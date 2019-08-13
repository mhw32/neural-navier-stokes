"""
Simulator for Two-Dimensional Incompressible Navier Stokes
using Direct Finite Difference Discretization. This means 
we must iteratively estimate pressure and impose somewhat
superficial boundary conditions on pressure. 
"""

import numpy as np
from tqdm import tqdm


class NavierStokesSystem():
    """
    Wrapper class around a 2D Incompressible Navier Stokes system.
    
    Args:
    -----
    u_ic : np.array
           initial conditions for u-momentum
    v_ic : np.array
           initial conditions for v-momentum
    p_ic : np.array
           initial conditions for pressure
    u_bc : list
           list of BoundaryCondition objects
    v_bc : list
           list of BoundaryCondition objects
    p_bc : list
           list of BoundaryCondition objects
    nt : integer
         number of time steps to run
    nit : integer
          number of internal iterations for convergence in pressure estimate
    nx : integer
         number of grid points across x
    ny : integer
         number of grid points across y
    dt : float
         discretized scale for time
    rho : float
          constant in the Navier Stokes equations
    nu : float
         constant in the Navier Stokes equations
    """

    def __init__(self, u_ic, v_ic, p_ic, u_bc, v_bc, p_bc, 
                 nt=200, nit=50, nx=50, ny=50, dt=0.001, rho=1, nu=0.1):
        super().__init__()
        self.u_ic, self.v_ic, self.p_ic = u_ic, v_ic, p_ic
        self.u_bc, self.v_bc, self.p_bc = u_bc, v_bc, p_bc
        self.nt, self.dt, self.nx, self.ny = nt, dt, nx, ny
        # hard code to size of x over 2 (un-dimensionalize to [-1, 1])
        self.dx, self.dy = 2. / (self.nx - 1), 2. / (self.ny - 1)
        self.nit, self.rho, self.nu = nit, rho, nu

    def _build_up_b(self, u, v):
        rho, dt, dx, dy = self.rho, self.dt, self.dx, self.dy
        b = np.zeros_like(u)
        b[1:-1, 1:-1] = (rho * (1 / dt * 
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + 
                         (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))) -
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                             (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)
        return b

    def _pressure_poisson(self, p, b):
        pn = np.empty_like(p)
        pn = p.copy()

        dt, dx, dy = self.dt, self.dx, self.dy
        rho, nu = self.rho, self.nu

        dx, dy = self.dx, self.dy
        for q in range(self.nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) -
                            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                            b[1:-1,1:-1])

            # set boundary conditions for pressure
            for bc in self.p_bc:
                p = bc.apply(p)
            
        return p

    def step(self, u, v, p):
        un, vn = u.copy(), v.copy()
        b = self._build_up_b(u, v)
        p = self._pressure_poisson(p, b)

        dt, dx, dy = self.dt, self.dx, self.dy
        rho, nu = self.rho, self.nu

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
        
        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # set boundary conditions
        for bc in self.u_bc:
            u = bc.apply(u)
        
        for bc in self.v_bc:
            v = bc.apply(v)

        return u, v, p

    def simulate(self):
        # collect propagations for dataset
        u_list, v_list, p_list = [], [], []
        u, v, p = self.u_ic, self.v_ic, self.p_ic
    
        for n in tqdm(range(self.nt)):
            u, v, p = self.step(u, v, p)
            u_list.append(u.copy())
            v_list.append(v.copy())
            p_list.append(p.copy())
    
        u_list = np.stack(u_list)
        v_list = np.stack(v_list)
        p_list = np.stack(p_list)

        return u_list, v_list, p_list


if __name__ == "__main__":
    from src.boundary import (DirichletBoundaryCondition, 
                              NeumannBoundaryCondition)

    nt  = 200
    nit = 50
    nx  = 50
    ny  = 50
    dt  = 0.001
    rho = 1
    nu  = 0.1

    dx = 2. / (nx - 1.)
    dy = 2. / (ny - 1.)

    u_ic = np.zeros((nx, ny))
    v_ic = np.zeros((nx, ny))
    p_ic = np.zeros((nx, ny))

    u_bc = [
        DirichletBoundaryCondition(0, 'left', dx, dy),
        DirichletBoundaryCondition(1, 'right', dx, dy),
        DirichletBoundaryCondition(0, 'top', dx, dy),
        DirichletBoundaryCondition(0, 'bottom', dx, dy),
    ]

    v_bc = [
        DirichletBoundaryCondition(0, 'left', dx, dy),
        DirichletBoundaryCondition(0, 'right', dx, dy),
        DirichletBoundaryCondition(0, 'top', dx, dy),
        DirichletBoundaryCondition(0, 'bottom', dx, dy),
    ]

    p_bc = [
        DirichletBoundaryCondition(0, 'top', dx, dy),
        NeumannBoundaryCondition(0, 'bottom', dx, dy),
        NeumannBoundaryCondition(0, 'left', dx, dy),
        NeumannBoundaryCondition(0, 'right', dx, dy),
    ]

    system = NavierStokesSystem(
        u_ic, v_ic, p_ic, u_bc, v_bc, p_bc, 
        nt=nt, nit=nit, nx=nx, ny=ny, dt=dt, 
        rho=rho, nu=nu,
    )

    u_data, v_data, p_data = system.simulate()
    np.savez('./data.npz', u=u_data, v=v_data, p=p_data)

