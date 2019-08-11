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

    def _predictor_step(self, u, v, u1, v1):
        dt, dx, dy = self.dt, self.dx, self.dy
        nu = self.nu

        # important to make copies
        ui, vi = u.copy(), v.copy()  # intermediate fields
        un, vn = u.copy(), v.copy()
        un1, vn1 = u1.copy(), v1.copy()  # u^{n-1}, v^{n-1}

        # Adam-Bashford for explicit momentum computation
        ui[1:-1, 1:-1] = un[1:-1, 1:-1] - dt * (3/2. * (un[1:-1, 1:-1] * (un[2:, 1:-1] - un[:-2, 1:-1]) / dx + 
                                                        vn[1:-1, 1:-1] * (un[2:, 1:-1] - un[:-2, 1:-1]) / dy)
                                               -1/2. * (un1[1:-1, 1:-1] * (un1[2:, 1:-1] - un1[:-2, 1:-1]) / dx + 
                                                        vn1[1:-1, 1:-1] * (un1[2:, 1:-1] - un1[:-2, 1:-1]) / dy))
                                        + dt * nu * (3/2. * ((un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dx**2 + 
                                                             (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dy**2)
                                                    -1/2. * ((un1[2:, 1:-1] - 2 * un1[1:-1, 1:-1] + un1[:-2, 1:-1]) / dx**2 + 
                                                             (un1[1:-1, 2:] - 2 * un1[1:-1, 1:-1] + un1[1:-1, :-2]) / dy**2))

        vi[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt * (3/2. * (un[1:-1, 1:-1] * (vn[2:, 1:-1] - vn[:-2, 1:-1]) / dx + 
                                                        vn[1:-1, 1:-1] * (vn[2:, 1:-1] - vn[:-2, 1:-1]) / dy)
                                               -1/2. * (un1[1:-1, 1:-1] * (vn1[2:, 1:-1] - vn1[:-2, 1:-1]) / dx + 
                                                        vn1[1:-1, 1:-1] * (vn1[2:, 1:-1] - vn1[:-2, 1:-1]) / dy))
                                        + dt * nu * (3/2. * ((vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dx**2 + 
                                                             (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dy**2)
                                                    -1/2. * ((vn1[2:, 1:-1] - 2 * vn1[1:-1, 1:-1] + vn1[:-2, 1:-1]) / dx**2 + 
                                                             (vn1[1:-1, 2:] - 2 * vn1[1:-1, 1:-1] + vn1[1:-1, :-2]) / dy**2))

        return ui, vi

    def _get_pressure(self, ui, vi, p):
        """
        Solve poisson pressure equation with successive over-relaxation (SOR).
        https://www3.nd.edu/~gtryggva/CFD-Course2010/2010-Lecture-11.pdf

        We need a separate process to solve
            laplace(p) = rho / dt * divergence(u*)
        
        Use successive over-relaxation to solve this elliptic eqn.
        """
        nx, ny = self.nx, self.ny
        dt, dx, dy = self.dt, self.dx, self.dy
        rho, beta = self.rho, self.beta

        tol, err, it = 5e-6, 1, 1
        pPrev = p.copy()

        C = np.zeros_like(ui)
        C[1:-1, 1:-1] = rho / dt * ((ui[2:, :] - ui[:-2, :]) / self.dx +
                                    (ui[:, 2:] - ui[:, :-2]) / self.dy)

        while ((err > tol) and (it < self.nit)):
            for i in range(1, nx):
                for j in range(1, ny):
                    p[i, j] =  (beta * 1/4. * ((p[i+1, j] + p[i-1, j]) / dx**2 + 
                                               (p[i, j+1] + p[i, j-1]) / dy**2 -
                                               C[i, j]) + 
                                (1 - beta) * p[i, j])

            err = np.max(np.abs(p - pPrev)) 
            pPrev = p.copy()
            it = it + 1

        return p

    def _correction_step(self, ui, vi, p):
        dt, dx, dy = self.dt, self.dx, self.dy
        un1 = ui - dt / dx * (p[2:, :] - p[:-2, :])
        vn1 = vi - dt / dy * (p[:, 2:] - p[:, :-2])
        
        return un1, vn1

    def step(self, un, vn, un1, vn1, p):
        ui, vi = self._predictor_step(un, vn, un1, vn1)

        # set boundary conditions
        for bc in self.u_bc:
            ui = bc.apply(ui)
        
        for bc in self.v_bc:
            vi = bc.apply(vi)

        p = self._get_pressure(ui, vi, p)

        # apply neumann boundary conditions
        for bc in self.p_bc:
            p = bc.apply(p)

        un1, vn1 = self._correction_step(ui, vi, p)
        return un1, vn1

    def simulate(self):
        # collect propagations for dataset
        u_list, v_list, p_list = [], [], []
        
        u, v, p = self.u_ic, self.v_ic, self.p_ic
        u1, v1 = self.u_ic.copy(), self.v_ic.copy()

        for n in tqdm(range(self.nt)):
            u, v, p = self.step(u, v, u1, v1, p)
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
    nx  = 50
    ny  = 50
    dt  = 0.001
    rho = 1
    nu  = 1

    dx = 2. / (nx - 1.)
    dy = 2. / (ny - 1.)

    u_ic = np.zeros((nx, ny))
    v_ic = np.zeros((nx, ny))
    p_ic = np.zeros((nx, ny))

    u_bc = [
        DirichletBoundaryCondition(0, 'left', dx, dy),
        DirichletBoundaryCondition(0, 'right', dx, dy),
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
        NeumannBoundaryCondition(0, 'top', dx, dy),
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
