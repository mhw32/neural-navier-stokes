import warnings
# raise errors on warnings otherwise hard to catch bugs
warnings.filterwarnings('error')

import numpy as np
from scipy.sparse import diags
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
    beta : float
           constant in successive over-relaxation
    """
    def __init__(self, u_ic, v_ic, p_ic, u_bc, v_bc, p_bc,
                 nt=200, nit=50, nx=50, ny=50, dt=0.001, 
                 rho=1, nu=1, beta=1.25):
        self.u_ic, self.v_ic, self.p_ic = u_ic, v_ic, p_ic
        self.u_bc, self.v_bc, self.p_bc = u_bc, v_bc, p_bc
        self.nt, self.nit, self.dt, self.nx, self.ny = nt, nit, dt, nx, ny
        # hard code to size of x over 2 (un-dimensionalize to [-1, 1])
        self.dx, self.dy = 2. / (self.nx - 1), 2. / (self.ny - 1)
        self.rho, self.nu, self.beta = rho, nu, beta

    def step(self, un, vn, un1, vn1, p):
        ui, vi = self._predictor_step(un, vn, un1, vn1)
        un1, vn1, p = self._correction_step(ui, vi, p)
        return un1, vn1, p

    def _predictor_step(self, un, vn, un1, vn1):
        """
        Parameters:
            un := u_n, vn := v_n, un1 := u_{n-1}, vn1 := v_{n-1}
            Computed velocity fields for current and last time step

        Returns:
            ui := u^*, vi := v^*
            Intermediary velocity fields
        """
        pass
    
    def _correction_step(ui, vi, p):
        """
        Parameters:
            ui := u^*, vi := v^*
            Intermediary velocity fields
            p 
            Pressure field for current time step
        
        Returns:
            un1 := u_{n+1}, vn1 := v_{n+1}, p1 := p_{n+1}
            Corrected velocity fields and new pressure field
        """
        pass

    def _init_variables(self):
        u, v, p = self.u_ic, self.v_ic, self.p_ic
        u, v, p = u.copy(), v.copy(), p.copy()

        for bc in self.u_bc:
            u = bc.apply(u)

        for bc in self.v_bc:
            v = bc.apply(v)

        for bc in self.p_bc:
            p = bc.apply(p)

        return u, v, p

    def simulate(self):
        u_list, v_list, p_list = [], [], []

        u, v, p = self._init_variables()
        u1, v1 = u.copy(), v.copy()

        for n in tqdm(range(self.nt)):
            _u, _v, p = self.step(u, v, u1, v1, p)
            u1, v1 = u.copy(), v.copy()
            u, v = _u.copy(), _v.copy()

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

    nt  = 200                 # number of timesteps
    nit = 200                 # number iterations for elliptic pressure eqn
    nx  = 50                  # size of spatial grid
    ny  = 50             
    dt  = 0.001         
    rho = 1                   # fluid density (kg / m^3)
    nu  = 0.1                 # fluid kinematic viscocity
    beta   = 1.25             # SOR hyperparameter

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
        rho=rho, nu=nu, beta=beta,
    )

    u_data, v_data, p_data = system.simulate()
    np.savez('./data.npz', u=u_data, v=v_data, p=p_data)
