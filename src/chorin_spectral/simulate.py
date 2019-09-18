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
    u_bc : list
           list of BoundaryCondition objects
    v_bc : list
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
    def __init__(self, u_ic, v_ic, p_ic, u_bc, v_bc, nt=200, nit=50,
                 nx=50, ny=50, dt=0.001, rho=1, nu=1, beta=1.25):
        self.u_ic, self.v_ic, self.p_ic = u_ic, v_ic, p_ic
        self.u_bc, self.v_bc = u_bc, v_bc  # no BC needed for pressure
        self.nt, self.nit, self.dt, self.nx, self.ny = nt, nit, dt, nx, ny
        # hard code to size of x over 2 (un-dimensionalize to [-1, 1])
        self.dx, self.dy = 2. / (self.nx - 1), 2. / (self.ny - 1)
        self.rho, self.nu, self.beta = rho, nu, beta

        # initialize a bunch of matrices
        self._pseudospectral_setup()

    def step(self, un, vn, un1, vn1, p):
        ui, vi = self._predictor_step(un, vn, un1, vn1)
        un1, vn1, p = self._correction_step(ui, vi, p)
        return un1, vn1, p

    def _pseudospectral_setup(self):
        Nx, Ny = self.nx, self.ny

        # -------------------------------------------
        # Precomputation for velocity Helmholtz step
        # -------------------------------------------

        # define a Gauss-Lobatto mesh by two vectors
        self.x_i = self._get_gauss_lobatto_points(Nx)
        self.y_i = self._get_gauss_lobatto_points(Ny)

        # get translation matrices
        self.Tx = self._get_T_matrix(Nx)
        self.Ty = self._get_T_matrix(Ny)
        self.Tx_inv = self._get_inv_T_matrix(Nx)
        self.Ty_inv = self._get_inv_T_matrix(Ny)

        # get derivative matrices
        self.Dx = self._get_D_matrix(Nx)
        self.Dy = self._get_D_matrix(Ny)
        self.Dx_sqr = self._get_D_sqr_matrix(Nx)
        self.Dy_sqr = self._get_D_sqr_matrix(Ny)

        # process boundary conditions
        (
            self.u_alpha_minus_x, self.u_alpha_plus_x,
            self.u_beta_minus_x, self.u_beta_plus_x,
            self.u_g_minus_x, self.u_g_plus_x,
            self.u_alpha_minus_y, self.u_alpha_plus_y,
            self.u_beta_minus_y, self.u_beta_plus_y,
            self.u_g_minus_y, self.u_g_plus_y,
        ) = self._process_boundary_conditions(self.u_bc)

        (
            self.v_alpha_minus_x, self.v_alpha_plus_x,
            self.v_beta_minus_x, self.v_beta_plus_x,
            self.v_g_minus_x, self.v_g_plus_x,
            self.v_alpha_minus_y, self.v_alpha_plus_y,
            self.v_beta_minus_y, self.v_beta_plus_y,
            self.v_g_minus_y, self.v_g_plus_y,
        ) = self._process_boundary_conditions(self.v_bc)


        def get_boundary_constants( D, N, alpha_minus, alpha_plus, beta_minus,
                                    beta_plus, g_minus, g_plus):
            """
            Helper function to compute usual quantities: e, c0-, c0+, b0j, bNj
                these are necessary to compute derivative metrices.
            """
            c0_minus = -beta_plus * D[0, N]
            c0_plus  = alpha_minus + beta_minus * D[N, N]
            cN_plus  = -beta_minus * D[N, 0]
            cN_minus = alpha_plus + beta_plus * D[0, 0]
            e = c0_plus * cN_minus - c0_minus * cN_plus

            # b0 and bN are N-2 length vectors (j=1 ... N-1)
            b0 = -c0_plus * beta_plus * D[0, 1:N] - c0_minus * beta_minus * D[N, 1:N]
            bN = -cN_minus * beta_minus * D[N, 1:N] - cN_plus * beta_plus * D[0, 1:N]

            return e, c0_minus, c0_plus, cN_minus, cN_plus, b0, bN

        (
            self.u_e_x,
            self.u_c0_minus_x, self.u_c0_plus_x,
            self.u_cN_minus_x, self.u_cN_plus_x,
            self.u_b0_x, self.u_bN_x,
        ) = get_boundary_constants( self.Dx, Nx,
                                    self.u_alpha_minus_x, self.u_alpha_plus_x,
                                    self.u_beta_minus_x, self.u_beta_plus_x,
                                    self.u_g_minus_x, self.u_g_plus_x )
        (
            self.u_e_y,
            self.u_c0_minus_y, self.u_c0_plus_y,
            self.u_cN_minus_y, self.u_cN_plus_y,
            self.u_b0_y, self.u_bN_y,
        ) = get_boundary_constants( self.Dy, Ny,
                                    self.u_alpha_minus_y, self.u_alpha_plus_y,
                                    self.u_beta_minus_y, self.u_beta_plus_y,
                                    self.u_g_minus_y, self.u_g_plus_y )
        (
            self.v_e_x,
            self.v_c0_minus_x, self.v_c0_plus_x,
            self.v_cN_minus_x, self.v_cN_plus_x,
            self.v_b0_x, self.v_bN_x,
        ) = get_boundary_constants( self.Dx, Nx,
                                    self.v_alpha_minus_x, self.v_alpha_plus_x,
                                    self.v_beta_minus_x, self.v_beta_plus_x,
                                    self.v_g_minus_x, self.v_g_plus_x )
        (
            self.v_e_y,
            self.v_c0_minus_y, self.v_c0_plus_y,
            self.v_cN_minus_y, self.v_cN_plus_y,
            self.v_b0_y, self.v_bN_y,
        ) = get_boundary_constants( self.Dy, Ny,
                                    self.v_alpha_minus_y, self.v_alpha_plus_y,
                                    self.v_beta_minus_y, self.v_beta_plus_y,
                                    self.v_g_minus_y, self.v_g_plus_y )

        # edit the derivative matrices to include boundary conditions
        # these are all N-2 x N-2 matrices
        u_Dx = self.Dx_sqr[1:Nx, 1:Nx] + 1./self.u_e_x * (self.u_b0_x * self.Dx_sqr[1:Nx, 0] +
                                                          self.u_bN_x * self.Dx_sqr[1:Nx, Nx])
        u_Dy = self.Dy_sqr[1:Ny, 1:Ny] + 1./self.u_e_y * (self.u_b0_y * self.Dy_sqr[1:Ny, 0] +
                                                          self.u_bN_y * self.Dy_sqr[1:Ny, Ny])
        v_Dx = self.Dx_sqr[1:Nx, 1:Nx] + 1./self.v_e_x * (self.v_b0_x * self.Dx_sqr[1:Nx, 0] +
                                                          self.v_bN_x * self.Dx_sqr[1:Nx, Nx])
        v_Dy = self.Dy_sqr[1:Ny, 1:Ny] + 1./self.v_e_y * (self.v_b0_y * self.Dy_sqr[1:Ny, 0] +
                                                          self.v_bN_y * self.Dy_sqr[1:Ny, Ny])

        # one time cost of computing eigenvalues and eigenvectors
        # these are very important to solving the Helmholtz equation
        #      which is what this step of Chorin's projection for NSE
        #      basically boils down to.
        # these again are all N-2 x N-2 matrices.
        # TODO: eigenvalues may be complex -- read to see what to do then
        self.u_Dx_lambda, self.u_Dx_P = np.linalg.eig(u_Dx)
        self.u_Dy_lambda, self.u_Dy_Q = np.linalg.eig(u_Dy)
        self.v_Dx_lambda, self.v_Dx_P = np.linalg.eig(v_Dx)
        self.v_Dy_lambda, self.v_Dy_Q = np.linalg.eig(v_Dy)

        # take inverses
        self.u_Dx_P_inv = np.linalg.inv(self.u_Dx_P)
        self.u_Dy_Q_inv = np.linalg.inv(self.u_Dy_Q)
        self.v_Dx_P_inv = np.linalg.inv(self.v_Dx_P)
        self.v_Dy_Q_inv = np.linalg.inv(self.v_Dy_Q)

        # -------------------------------------------
        # Precomputation for pressure projection step
        # -------------------------------------------

        # get derivative matrices for pressure
        self.DPx = self._get_D_matrix_degrees_minus_2(Nx)
        self.DPy = self._get_D_matrix_degrees_minus_2(Ny)

        self.DxDPx = self.Dx[1:-1,1:-1] @ self.DPx
        self.DyDPy = self.Dy[1:-1,1:-1] @ self.DPy

        self.DxDPx_lambda, self.DxDPx_P = np.linalg.eig(self.DxDPx)
        self.DyDPy_lambda, self.DyDPy_Q = np.linalg.eig(self.DyDPy)
        self.DxDPx_P_inv = np.linalg.inv(self.DxDPx_P)
        self.DyDPy_Q_inv = np.linalg.inv(self.DyDPy_Q)

    def _process_boundary_conditions(self, bc_list):
        for bc in bc_list:
            if bc.type == 'dirichlet':
                if bc.boundary == 'left':
                    alpha_minus_x = 1
                    g_minus_x = bc.value
                elif bc.boundary == 'right':
                    alpha_plus_x = 1
                    g_plus_x = bc.value
                elif bc.boundary == 'top':
                    alpha_minus_y = 1
                    g_minus_y = bc.value
                elif bc.boundary == 'bottom':
                    alpha_plus_y = 1
                    g_plus_y = bc.value
                else:
                    raise Exception('Boundary side {} not supported'.format(bc.boundary))
            elif bc.type == 'neumann':
                # we don't support Neumann boundary conditions
                # yet... but no barriers to doing so... just work.
                raise NotImplementedError
            else:
                raise Exception('Boundary type {} not supported'.format(bc.type))

        # bc we dont support Neumann BC yet
        beta_minus_x, beta_plus_x = 0, 0
        beta_minus_y, beta_plus_y = 0, 0

        return  alpha_minus_x, alpha_plus_x, beta_minus_x, beta_plus_x, g_minus_x, g_plus_x, \
                alpha_minus_y, alpha_plus_y, beta_minus_y, beta_plus_y, g_minus_y, g_plus_y

    def _predictor_step(self, un, vn, un1, vn1):
        """
        Parameters:
            un := u_n, vn := v_n, un1 := u_{n-1}, vn1 := v_{n-1}
            Computed velocity fields for current and last time step

        Returns:
            ui := u^*, vi := v^*
            Intermediary velocity fields
        """
        # pg 78 Peyret: Spectral methods for Incompressible Viscious Flow
        Nx, Ny = self.nx, self.ny

        def get_boundary_values(un, g_minus_x, g_plus_x, g_minus_y, g_plus_y,
                                e_x, c0_minus_x, c0_plus_x, cN_minus_x, cN_plus_x, b0_x, bN_x,
                                e_y, c0_minus_y, c0_plus_y, cN_minus_y, cN_plus_y, b0_y, bN_y):
            # returns VECTORS of size N-2 for each boundary row and column
            un_x0 = 1./e_x * np.sum(b0_x[:, np.newaxis] * un, axis=0) + \
                    1./e_x * (c0_minus_x * g_minus_x + c0_plus_x * g_plus_x)
            un_xN = 1./e_x * np.sum(bN_x[:, np.newaxis] * un, axis=0)
            un_y0 = 1./e_y * np.sum(b0_y[np.newaxis, :] * un, axis=1) + \
                    1./e_y * (c0_minus_y * g_minus_y + c0_plus_y * g_plus_y)
            un_yN = 1./e_y * np.sum(bN_y[np.newaxis, :] * un, axis=1)

            return un_x0, un_xN, un_y0, un_yN

        # compute the RHS of the linear system (F)
        # 2u* - \bigtriangleup t \Delta u^* = F
        #   use the derivative matrices whenever we need to compute derivatives

        _un, _un1 = un[1:Nx, 1:Ny], un1[1:Nx, 1:Ny]
        _vn, _vn1 = vn[1:Nx, 1:Ny], vn1[1:Nx, 1:Ny]

        _un_dx, _un_dy = self.Dx[1:-1, 1:-1] @ _un, _un @ self.Dy[1:-1, 1:-1].T
        _un1_dx, _un1_dy = self.Dx[1:-1, 1:-1] @ _un1, _un1 @ self.Dy[1:-1, 1:-1].T

        _vn_dx, _vn_dy = self.Dx[1:-1, 1:-1] @ _vn, _vn @ self.Dy[1:-1, 1:-1].T
        _vn1_dx, _vn1_dy = self.Dx[1:-1, 1:-1] @ _vn1, _vn1 @ self.Dy[1:-1, 1:-1].T

        _un_ddx, _un_ddy = self.Dx_sqr[1:-1, 1:-1] @ _un, _un @ self.Dy_sqr[1:-1, 1:-1].T
        _un1_ddx, _un1_ddy = self.Dx_sqr[1:-1, 1:-1] @ _un1, _un1 @ self.Dy_sqr[1:-1, 1:-1].T

        _vn_ddx, _vn_ddy = self.Dx_sqr[1:-1, 1:-1] @ _vn, _vn @ self.Dy_sqr[1:-1, 1:-1].T
        _vn1_ddx, _vn1_ddy = self.Dx_sqr[1:-1, 1:-1] @ _vn1, _vn1 @ self.Dy_sqr[1:-1, 1:-1].T

        # u_F and v_F are both N-2 x N-2 matrices
        u_F = 2 * _un - 3 * self.dt * (_un * _un_dx + _vn * _un_dy) + \
                self.dt * (_un1 * _un1_dx + _vn1 * _un1_dy) + \
                self.dt * (_un_ddx + _un_ddy)
        v_F = 2 * _vn - 3 * self.dt * (_un * _vn_dx + _vn * _vn_dy) + \
                self.dt * (_un1 * _vn1_dx + _vn1 * _vn1_dy) + \
                self.dt * (_vn_ddx + _vn_ddy)

        # solve linear system for u first
        u_H_tilde = self.u_Dx_P_inv @ u_F
        u_H_hat   = u_H_tilde @ self.u_Dy_Q_inv.T
        u_hat     = u_H_hat / ( 2. - self.dt * dup_vector_by_row(self.u_Dx_lambda, Nx - 1) -
                                self.dt * dup_vector_by_col(self.u_Dy_lambda, Ny - 1) )
        u_tilde   = u_hat @ self.u_Dy_Q.T
        u_soln    = self.u_Dx_P @ u_tilde

        # repeat for v -- 4 matrix multplications
        v_H_tilde = self.v_Dx_P_inv @ v_F
        v_H_hat   = v_H_tilde @ self.v_Dy_Q_inv.T
        v_hat     = v_H_hat / ( 2. - self.dt * dup_vector_by_row(self.v_Dx_lambda, Nx - 1) -
                                self.dt * dup_vector_by_col(self.v_Dy_lambda, Ny - 1) )
        v_tilde   = v_hat @ self.v_Dy_Q.T
        v_soln    = self.v_Dx_P @ v_tilde

        # impose boundary conditions
        u_soln_x0, u_soln_xN, u_soln_y0, u_soln_yN = \
            get_boundary_values(
                u_soln,
                self.u_g_minus_x, self.u_g_plus_x, self.u_g_minus_y, self.u_g_plus_y,
                self.u_e_x, self.u_c0_minus_x, self.u_c0_plus_x,
                self.u_cN_minus_x, self.u_cN_plus_x, self.u_b0_x, self.u_bN_x,
                self.u_e_y, self.u_c0_minus_y, self.u_c0_plus_y,
                self.u_cN_minus_y, self.u_cN_plus_y, self.u_b0_y, self.u_bN_y,
            )
        v_soln_x0, v_soln_xN, v_soln_y0, v_soln_yN = \
            get_boundary_values(
                v_soln,
                self.v_g_minus_x, self.v_g_plus_x, self.v_g_minus_y, self.v_g_plus_y,
                self.v_e_x, self.v_c0_minus_x, self.v_c0_plus_x,
                self.v_cN_minus_x, self.v_cN_plus_x, self.v_b0_x, self.v_bN_x,
                self.v_e_y, self.v_c0_minus_y, self.v_c0_plus_y,
                self.v_cN_minus_y, self.v_cN_plus_y, self.v_b0_y, self.v_bN_y,
            )

        # put it all together
        # TODO: the corners are ignored... fix?
        u_intermediate = np.zeros((Nx + 1, Ny + 1))
        u_intermediate[1:Nx, 1:Ny] = u_soln
        u_intermediate[0, 1:Ny] = u_soln_x0
        u_intermediate[Nx, 1:Ny] = u_soln_xN
        u_intermediate[1:Nx, 0] = u_soln_y0
        u_intermediate[1:Nx, Ny] = u_soln_yN

        v_intermediate = np.zeros((Nx + 1, Ny + 1))
        v_intermediate[1:Nx, 1:Ny] = v_soln
        v_intermediate[0, 1:Ny] = v_soln_x0
        v_intermediate[Nx, 1:Ny] = v_soln_xN
        v_intermediate[1:Nx, 0] = v_soln_y0
        v_intermediate[1:Nx, Ny] = v_soln_yN

        return u_intermediate, v_intermediate

    def _correction_step(self, ui, vi, p):
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
        Nx, Ny = self.nx, self.ny
        # step 1: get boundary values and build u_tau and v_tau
        u_tau = np.stack([np.ones(Ny - 1) * self.u_g_minus_x,
                          np.ones(Ny - 1) * self.u_g_plus_x])
        v_tau = np.stack([np.ones(Nx - 1) * self.v_g_minus_y,
                          np.ones(Nx - 1) * self.v_g_plus_y]).T
        # these two are probably the same 
        Dx_bar = np.stack([self.Dx[1:Nx, 0], self.Dx[1:Nx, Nx]]).T
        Dy_bar = np.stack([self.Dy[1:Ny, 0], self.Dy[1:Ny, Ny]]).T

        S = -(Dx_bar @ u_tau + v_tau @ Dy_bar.T)
        
        # now we need to solve the following Uzawa type matrix
        # Dx\hat{Dx} Q + Q(Dy\hat{D}y)^T = -\sigma(S - Dx\tilde{U} - \tilde{V}Dy^T)

        # first lets compute the right hand side!
        H = -self.rho / self.dt * (S - self.Dx[1:-1, 1:-1] @ ui[1:-1, 1:-1] - vi[1:-1, 1:-1] @ self.Dy[1:-1, 1:-1].T)

        # do the matrix multiplication trick
        H_tilde = self.DxDPx_P_inv @ H
        H_hat   = H_tilde @ self.DyDPy_Q_inv.T
        Q_hat   = H_hat / ( dup_vector_by_row(self.DxDPx_lambda, Nx - 1) + 
                            dup_vector_by_col(self.DyDPy_lambda, Ny - 1) )
        Q_tilde = Q_hat @ self.DyDPy_Q.T
        Q       = self.DxDPx_P @ Q_tilde

        # transform this back into U and V space
        u_np1, v_np1 = ui.copy(), vi.copy()
        u_np1[1:-1,1:-1] = u_np1[1:-1,1:-1] - self.DxDPx @ Q * self.dt / self.rho
        v_np1[1:-1,1:-1] = v_np1[1:-1,1:-1] - Q @ self.DyDPy.T * self.dt / self.rho

        return u_np1, v_np1

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
        inv_T = np.stack([  self._get_gauss_lobatto_points(N, k=k)
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
                    diff = 2 * np.sin((j + i) * np.pi / (2. * N)) * \
                            np.sin((j - i) * np.pi / (2. * N))
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
        D_sqr_tmp = D @ D.T  # FIXME: check this
        D_sqr = np.zeros_like(D_sqr_tmp)

        for i in range(0, N + 1):
            for j in range(0, N + 1):
                D_sqr[i, j] = D_sqr_tmp[i, j]

        for i in range(0, N + 1):
            # we can include when i == j in the sum bc its 0
            D_sqr[i, i] = -np.sum(D_sqr[i, :])

        return D_sqr

    def _get_D_matrix_degrees_minus_2(self, N):
        """
        This is used for the P_N - P_{N-2} projection trick: 
            Velocity is approximated using Chebyshev polynomials 
            of degree N but Pressure is approximated using degree
            N-2; however, both approximations use the same 
            Gauss-Lobatto points -- this requires us to make a 
            new differentiation matrix.

        We do not use the tricks for numerical stability here -- ironically
        for PN-2, they introduce instability.
        
        We also compute the diagonals last to ensure proper summation.
        """
        D = np.zeros((N + 1, N + 1))
        x = self._get_gauss_lobatto_points(N, k=1)
        
        for i in range(1, N):
            for j in range(1, N):
                if i != j:
                    D[i, j] = ((-1)**(j+1) * (1. - x[j]**2) / ((1. - x[i]**2) * (x[i] - x[j])))
                else:
                    D[i, i] = 3*x[i] / (2. * (1. - x[i]**2)) 
        
        D = D[1:-1, 1:-1]  # only defined for i,j=1...N-1
        return D

    # --- end section on pseudospectral method helpers

    def _init_variables(self):
        u, v, p = self.u_ic, self.v_ic, self.p_ic
        u, v, p = u.copy(), v.copy(), p.copy()

        for bc in self.u_bc:
            u = bc.apply(u)

        for bc in self.v_bc:
            v = bc.apply(v)

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


def dup_vector_by_row(v, n):
    return v[:, np.newaxis].repeat(n, axis=1)

def dup_vector_by_col(v, n):
    return dup_vector_by_row(v, n).T


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

    system = NavierStokesSystem(
        u_ic, v_ic, p_ic, u_bc, v_bc,
        nt=nt, nit=nit, nx=nx, ny=ny, dt=dt,
        rho=rho, nu=nu, beta=beta,
    )

    u_data, v_data, p_data = system.simulate()
    np.savez('./data.npz', u=u_data, v=v_data, p=p_data)
