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


class BoundaryCondition():
    """General class for a boundary condition.
    
    Args:
    -----
    x := integer 
         x-coordinate in matrix A
    y := integer 
         y-coordinate in matrix A
    """

    def __init__(self, x, y, min_x, max_x, min_y, max_y, 
                 dirichlet=None, neumann=None, periodic=False):
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
        self.x, self.x_min, self.x_max = x, x_min, x_max
        self.y, self.y_min, self.y_may = y, y_min, y_max

    def is_dirichlet(self):
        return self.dirichlet is not None

    def is_neumann(self):
        return self.neumann is not None

    def is_periodic(self):
        return self.periodic

    def is_left_x_boundary(self):
        return self.x == self.min_x
    
    def is_right_x_boundary(self):
        return self.x == self.max_x
    
    def is_left_y_boundary(self):
        return self.y == self.min_y
    
    def is_right_y_boundary(self):
        return self.y == self.max_y

    def apply(self):
        raise NotImplementedError



class UMomentumBoundaryCondition(BoundaryCondition):
    
    def _apply_dirichlet(self, u):
        """
        Dirichlet BCs are easy, all we have to do is set 
        the point to a particular value.
        """
        u[self.x, self.y] = self.dirichlet

    def _apply_neumann(self, u, dunit):
        """
        Assume that we now du/dx at some coordinate (i,j). Then, 
        we can say as a simple forward difference estimate.
            
            du/dx|(i,j) = (-3*u[i,j] + 4u[i+1,j] - u[i+2,j]) / 2*dx
            du/dy|(i,j) = (-3*u[i,j] + 4u[i,j+1] - u[i,j+2]) / 2*dy
        
        For end points on the other edge, we would use a backward 
        difference estimate.
            
            du/dx|(i,j) = (3*u[i,j] - 4u[i-1,j] + u[i-2,j]) / 2*dx
            du/dy|(i,j) = (3*u[i,j] - 4u[i,j-1] + u[i,j-2]) / 2*dy
        
        More tricks exist for central difference that include making
        ghost mesh points but too complicated.

        Args:
        -----
        u := np.array
             momentum tensor
        dunit := discrete interval along x/y direction
        """
        if self.is_left_x_boundary():
            dudx, dx = self.neumann, dunit
            u[self.x, self.y] = (4./3. * u[self.x+1,self.y] - 
                                 1./3 * u[self.x+2,self.y] - 
                                 2./3. * dx  * dudx)
        elif self.is_right_x_boundary():
            dudx, dx = self.neumann, dunit
            u[self.x, self.y] = (4./3. * u[self.x-1,self.y] - 
                                 1./3 * u[self.x-2,self.y] + 
                                 2./3. * dx  * dudx)
        elif self.is_left_y_boundary():
            dudy, dy = self.neumann, dunit
            u[self.x, self.y] = (4./3. * u[self.x,self.y+1] - 
                                 1./3 * u[self.x,self.y+2] - 
                                 2./3. * dy  * dudy)
        elif self.is_right_y_boundary():
            dudy, dy = self.neumann, dunit
            u[self.x, self.y] = (4./3. * u[self.x,self.y-1] - 
                                 1./3 * u[self.x,self.y-2] + 
                                 2./3. * dy  * dudy)
        else:
            raise Exception('invalid point not on boundary.')

    def _apply_periodic(self):
        raise NotImplementedError


class VMomentumBoundaryCondition(UMomentumBoundaryCondition):
    # these will become very different once we start to 
    # think about periodic conditions.
    pass


class PressureBoundaryCondition(UMomentumBoundaryCondition):
    pass
