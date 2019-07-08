"""
Master function for generating a 2D Navier Stokes
System. We will randomly bit initial conditions 
and boundary conditions. 

Optionally, users can optionally add variance in 
a source term as well as add periodic boundary 
conditions.

TODO: handle Neumann boundary conditions.
"""

import numpy as np


class BoundaryCondition():
    """General class for a boundary condition."""

    def __init__(self, dirichlet=None, neumann=None, periodic=False):
        assert (dirichlet not None) or (neumann not None) or (not periodic)
        if dirichlet is not None:
            assert neumann is None and not periodic
        if neumann is not None:
            assert dirichlet is None and not periodic
        if periodic:
            assert neumann is None and dirichlet is None
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.periodic = periodic

    def get(self):
        if self.dirichlet:
            return self.dirichlet, 'dirichlet'
        elif self.neumann:
            return self.neumann, 'neumann'
        else:
            return self.periodic, 'periodic'


class BoundaryConditions():
    """Specifies boundary conditions for either Momentum 
    or Pressure (in Navier Stokes).

    You can either specify a dirichlet, neumann, or periodic
    condition at each boundary. We will not allow multiple 
    types of boundaries.

    Args:
    -----
    x0 := BoundaryCondition object
          boundary condition at x=0
    y0 := BoundaryCondition object
          boundary condition at y=0
    xn := BoundaryCondition object
          boundary condition at x=n
    yn := BoundaryCondition object 
          boundary condition at y=n
    """
    def __init__(self, x0_bc, y0_bc, xn_bc, yn_bc):
        self.x0_bc = x0_bc
        self.y0_bc = y0_bc
        self.xn_bc = xn_bc
        self.yn_bc = yn_bc

    def is_dirichlet(self):
        pass
    
    def is_neumann(self):
        pass
    
    def is_periodic(self):


class PressureBoundaryConditions(BoundaryConditions):
    pass


class MomentumBoundaryConditions(BoundaryConditions):
    pass
