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


class BoundaryConditions():
    """Specifies boundary conditions for either Momentum 
    or Pressure (in Navier Stokes). As a user, you can either
    specify Dirichlet BCs (x0, y0, xn, yn) or Neumann BCs 
    (dx0, dy0, dxn, dyn). 

    Args:
    -----
    x0 := float/string 
          value of momentum/pressure at x=0
    y0 := float/string
          value of momentum/pressure at y=0
    xn := float/string
          value of momentum/pressure at x=n
    yn := float/string 
          value of momentum/pressure at y=n
    """
    def __init__(self, x0=None, y0=None, xn=None, yn=None,
                 dx0=None, dy0=None, dxn=None, dyn=None):
        self.condition = {'x0': x0, 'y0': y0, 'xn': xn, 'yn': yn,
                          'dx0': dx0, 'dy0': dy0, 'dxn': dxn, 'dyn': dyn}
        assert 


class PressureBoundaryConditions(BoundaryConditions):
    pass


class MomentumBoundaryConditions(BoundaryConditions):
    pass
