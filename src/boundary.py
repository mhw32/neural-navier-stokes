class BaseBoundaryCondition(object):
    """
    General class for a boundary condition.

    Args:
    -----
    value := float
             discrete or derivative value
    boundary := string
                'bottom' (y=0), 'top' (y=N), 'left' (x=0), 'right' (x=N)
    dx, dy := integer
              discretization size for derivation
    """
    def __init__(self, value, boundary, dx, dy):
        super().__init__()
        assert isinstance(boundary, str)
        assert isinstance(dx, float)
        assert isinstance(dy, float)
        assert boundary in ['left', 'right', 'bottom', 'top']

        self.value = value
        self.boundary = boundary
        self.dx, self.dy = dx, dy

    def apply(self, A):
        raise NotImplementedError


class DirichletBoundaryCondition(BaseBoundaryCondition):
    def __init__(self, value, boundary, dx, dy):
        super().__init__(value, boundary, dx, dy)
        self.type = 'dirichlet'

    def apply(self, A):
        """
        Dirichlet BCs are easy, all we have to do is set
        the point to a particular value.
        """
        if self.boundary == 'left':
            A[0, :] = self.value
        elif self.boundary == 'right':
            A[-1, :] = self.value
        elif self.boundary == 'bottom':
            A[:, 0] = self.value
        elif self.boundary == 'top':
            A[:, -1] = self.value

        return A


class NeumannBoundaryCondition(BaseBoundaryCondition):
    def __init__(self, value, boundary, dx, dy):
        super().__init__(value, boundary, dx, dy)
        self.type = 'neumann'

    def apply(self, A):
        """
        Assume that we now du/dx at some coordinate (i,j). Then,
        we can say as a simple forward difference estimate.

            du/dx|(i,j) = u[i+1,j] - u[i,j] / dx
            du/dy|(i,j) = u[i,j+1] - u[i,j] / dy

        For end points on the other edge, we would use a backward
        difference estimate.

            du/dx|(i,j) = u[i,j] - u[i-1,j] / dx
            du/dy|(i,j) = u[i,j] - u[i,j-1] / dy

        More tricks exist for central difference that include making
        ghost mesh points but too complicated.
        """
        if self.boundary == 'left':
            dAdx = self.value
            A[0, :] = A[1, :] - self.dx * dAdx
        elif self.boundary == 'right':
            dAdx = self.value
            A[-1, :] = A[-2, :] + self.dx * dAdx
        elif self.boundary == 'bottom':
            dAdy = self.value
            A[:, 0] = A[:, 1] - self.dy * dAdy
        elif self.boundary == 'top':
            dAdy = self.value
            A[:, -1] = A[:, -2] + self.dy * dAdy

        return A
