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

    # --- begin section on pseudospectral method helpers

    def _get_c_k(self, k):
        assert k >= 0
        return 2 if k == 0 else 1
    
    def _get_bar_c_k(self, k, N):
        assert k >= 0
        return 2 if (k == 0 or k == N) else 1

    def _get_gauss_lobatto_points(self, N, k=1):
        # N => number of points
        i = np.arange(N + 1)
        x_i = np.cos(k * np.pi * i / float(N))
        return x_i

    def _get_T_matrix(self, N):
        """
        Matrix to convert back and forth between spectral coefficients, 
        \hat{u}_k, and the values at the collocation points, u_N(x_i).
        This is just a matrix multiplication.

        \mathcal{T} = [\cos k\pi i / N], k,i = 0, ..., N
        \mathcal{U} = \mathcal{T}\hat{\mathcal{U}}

        where \mathcal{U} = [u(x_0), ..., u(x_N)], the values of the 
        function at the coordinate points.
        """
        T = [
            self._get_gauss_lobatto_points(N, k=k)
            for k in np.arange(0, N + 1)
        ]
        # N(k) x N(i) since this will be multiplied by the matrix
        # of spectral coefficients (k)
        return np.stack(T)

    def _get_inv_T_matrix(self, N):
        """
        \mathcal{T}^{-1} = [2(\cos \pi i / N)/(\bar{c}_k \bar{c}_i N)]
        \hat{\mathcal{U}} = \mathcal{T}\mathcal{U}

        where \hat{\mathcal{U}} = [\hat{u}_0, ..., \hat{u}_N], the 
        coefficients of the truncated spectral approximation.
        """
        inv_T = np.stack([  self._get_gauss_lobatto_points(N, k=1)
                            for k in np.arange(0, N + 1)  ])
        inv_T = inv_T.T  # size N(i) x N(k)
        
        # bar_c_i is size N(i) x N(k)
        bar_c_i = np.stack([np.repeat(self._get_bar_c_k(i, N), N + 1)
                            for i in np.arange(0, N + 1)])
        bar_c_k = bar_c_i.T

        inv_T = 2 * inv_T / (bar_c_k * bar_c_i * N)
        # N(i) x N(k) since this will be multiplied by the matrix
        # of coordinate values
        return inv_T

    def _get_D_matrix(self, N):
        """
        Matrix to compute derivative of coordinate values.
            \mathcal{D} = [d^{(1)}_{i,j}] for i,j=0...N
        where if 0 <= i,j <= N, i != j
            d^{(1)}_{i,j} = \frac{\bar{c}_i}{\bar{c}_j} \frac{(-1)^{i+j}}{(x_i - x_j}
        else if i <= i <= N - 1, i == j
            d^{(1)}_{i,i} = -\frac{x_i}{2(1 - x_i^2)}
        else
            d^{(1)}_{0,0} = -d^{(1)}_{N, N}
                          = \frac{2N^2 + 1}{6}

        For numerical stability:
            let x_i - x_j = 2\sin(\frac{(j+i)\pi}{2N})\sin(\frac{(j-i)\pi}{2N})
            and 1 - x_i^2 = \sin^2(i\pi/N)

        To ensure the matrix sums to a constant:
            d^{(1)}_{i,i} = -\sum_{j=0,j!=i}^N d^{(1)}_{i,j}, i=0...N

        This will be used such that 
            \mathcal{U}^{(1)} = \mathcal{D}\mathcal{U}
        """
        # kind of slow to loop but we only ever have to do this once
        D = np.zeros((N + 1, N + 1))
        for i in range(0, N + 1):
            for j in range(0, N + 1):
                if i != j:
                    bar_c_i = self._get_bar_c_k(i, N)
                    bar_c_j = self._get_bar_c_k(j, N)
                    diff = 2 * np.sin((j + i) * np.pi / (2 * N)) * \
                            np.sin((j - i) * np.pi / (2 * N))
                    D[i, j] = bar_c_i / bar_c_j * (-1)**(i + j) / diff

        # now we fill out the diagonals
        for i in range(0, N + 1):
            # we can include when i == j in the sum bc its 0
            D[i, i] = -np.sum(D[i, :])
        
        return D

    def _get_D_sqr_matrix(self, N):
        """
        A second matrix to compute second derivatives of 
        coordinate values. In practice, we can just square 
        the first derivative matrix and apply some fixes.

        This will be used such that 
            \mathcal{U}^{(2)} = \mathcal{D}^2\mathcal{U}
        """
        D = self._get_D_matrix(N)
        D_sqr_tmp = D**2
        D_sqr = np.zeros_like(D_sqr_tmp)

        for i in range(0, N + 1):
            for j in range(0, N + 1):
                D_sqr[i, j] = D_sqr_tmp[i, j]
        
        for i in range(0, N + 1):
            # we can include when i == j in the sum bc its 0
            D_sqr[i, i] = -np.sum(D_sqr[i, :])
        
        return D_sqr

    # --- end section on pseudospectral method helpers

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
