"""
Master function for generating a 2D Navier Stokes
System. We will randomly bit initial conditions 
and boundary conditions. 

Optionally, users can optionally add variance in 
a source term as well as add periodic boundary 
conditions.

Notes:
    Dirichlet boundary conditions:
        https://github.com/barbagroup/CFDPython

    Neumann boundary conditions: 
        http://folk.ntnu.no/leifh/teaching/tkt4140/._main056.html
        
    Periodic boundary conditions:
        https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb
"""

import numpy as np
from tqdm import tqdm


class BaseBoundaryCondition(object):
    """General class for a boundary condition.
    
    Args:
    -----
    x := integer 
         x-coordinate in matrix A
    y := integer 
         y-coordinate in matrix A
    """

    def __init__(self, i, j, min_i, max_i, min_j, max_j, dx, dy, 
                 dirichlet=None, neumann=None, periodic=False):
        super().__init__()
        assert (dirichlet is not None) or (neumann is not None) or (not periodic)
        if dirichlet is not None:
            assert neumann is None and not periodic
        if neumann is not None:
            assert dirichlet is None and not periodic
        if periodic:
            assert neumann is None and dirichlet is None
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.periodic = periodic
        self.i, self.min_i, self.max_i = i, min_i, max_i
        self.j, self.min_j, self.max_j = j, min_j, max_j
        self.dx, self.dy = dx, dy

    def is_dirichlet(self):
        return self.dirichlet is not None

    def is_neumann(self):
        return self.neumann is not None

    def is_periodic(self):
        return self.periodic

    def is_left_x_boundary(self):
        return self.i == self.min_i
    
    def is_right_x_boundary(self):
        return self.i == self.max_i
    
    def is_left_y_boundary(self):
        return self.j == self.min_j
    
    def is_right_y_boundary(self):
        return self.j == self.max_j

    def apply(self):
        raise NotImplementedError


class BoundaryCondition(BaseBoundaryCondition):
    
    def apply(self, A):
        if self.is_dirichlet():
            return self._apply_dirichlet(A)
        elif self.is_neumann():
            return self._apply_neumann(A)
        elif self._is_periodic():
            return self._apply_periodic(A)

    def _apply_dirichlet(self, A):
        """
        Dirichlet BCs are easy, all we have to do is set 
        the point to a particular value.
        """
        A[self.i, self.j] = self.dirichlet
        return A

    def _apply_neumann(self, A):
        """
        Assume that we now du/dx at some coordinate (i,j). Then, 
        we can say as a simple forward difference estimate.
            
            du/dx|(i,j) = (-3*A[i,j] + 4A[i+1,j] - A[i+2,j]) / 2*dx
            du/dy|(i,j) = (-3*u[i,j] + 4u[i,j+1] - u[i,j+2]) / 2*dy
        
        For end points on the other edge, we would use a backward 
        difference estimate.
            
            du/dx|(i,j) = (3*u[i,j] - 4u[i-1,j] + u[i-2,j]) / 2*dx
            du/dy|(i,j) = (3*u[i,j] - 4u[i,j-1] + u[i,j-2]) / 2*dy
        
        More tricks exist for central difference that include making
        ghost mesh points but too complicated.
        """
        if self.is_left_x_boundary():
            dAdx = self.neumann
            A[self.i, self.j] = (4./3. * A[self.i+1,self.j] - 
                                 1./3 * A[self.i+2,self.j] - 
                                 2./3. * self.dx * dAdx)
        elif self.is_right_x_boundary():
            dAdx = self.neumann
            A[self.i, self.j] = (4./3. * A[self.i-1,self.j] - 
                                 1./3 * A[self.i-2,self.j] + 
                                 2./3. * self.dx * dAdx)
        elif self.is_left_y_boundary():
            dAdy = self.neumann
            A[self.i, self.j] = (4./3. * A[self.i,self.j+1] - 
                                 1./3 * A[self.i,self.j+2] - 
                                 2./3. * self.dy * dAdy)
        elif self.is_right_y_boundary():
            dAdy = self.neumann
            A[self.i, self.j] = (4./3. * A[self.i,self.j-1] - 
                                 1./3 * A[self.i,self.j-2] + 
                                 2./3. * self.dy * dAdy)
        else:
            raise Exception('invalid point not on boundary.')

        return A

    def _apply_periodic(self, A):
        raise NotImplementedError


class MomentumBoundaryCondition(BoundaryCondition):
    # these will become very different once we start to 
    # think about periodic conditions.
    pass


class PressureBoundaryCondition(BoundaryCondition):
    pass


class NavierStokesSystem(object):
    """Wrapper class around a Navier Stokes system.
    
    Args:
    -----
    u_ic : np.array
           initial conditions for u-momentum
    v_ic : np.array
               initial conditions for v-momentum
    p_ic : np.array
               initial conditions for pressure
    u_bc : list
           list of MomentumBoundaryCondition objects
    v_bc : list
           list of MomentumBoundaryCondition objects
    p_bc : list
           list of PressureBoundaryCondition objects
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
    F : float
        constant representing source flow
    """

    def __init__(self, u_ic, v_ic, p_ic, u_bc, v_bc, p_bc, 
                 nt=200, nit=50, nx=50, ny=50, dt=0.001, rho=1, nu=.1, F=0):
        super().__init__()
        self.u_ic, self.v_ic, self.p_ic = u_ic.copy(), v_ic.copy(), p_ic.copy()
        self.u_bc, self.v_bc, self.p_bc = u_bc, v_bc, p_bc
        self.nt, self.dt, self.nx, self.ny = nt, dt, nx, ny
        self.dx, self.dy = 2. / (self.nx - 1), 2. / (self.ny - 1)
        self.nit, self.rho, self.nu, self.F = nit, rho, nu, F

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
        rho, nu, F = self.rho, self.nu, self.F

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                         F * dt)
        
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
