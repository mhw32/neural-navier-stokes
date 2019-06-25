"""Runge Kutta Solver (RK4) for Lorenz system. In particular,
you can vary the hyperparameter `h`."""

import numpy as np

import matplotlib as mpl
mpl.switch_backend('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz_ode_compute(n, t_final, x0, y0, z0):
    """
    Returns
    -------
    Output, real T(N+1), X(N+1), Y(N+1), Z(N+1), the T, X, Y, and Z values 
    of the discrete solution.
    """
    dt = t_final / n
    t = np.linspace(0.0, t_final, n + 1)

    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)

    # initial conditions
    x[0] = 8.0
    y[0] = 1.0
    z[0] = 1.0

    for i in range(n):
        xyz = np.array([x[i], y[i], z[i]])
        xyz = rk4vec(t[i], 3, xyz, dt, lorenz_rhs)

        x[i+1] = xyz[0]
        y[i+1] = xyz[1]
        z[i+1] = xyz[2]

    return t, x, y, z


def lorenz_rhs(t, m, xyz):
    """
    Three ordinary differential equations that 
    govern the Lorenz equations: 
    
    https://en.wikipedia.org/wiki/Lorenz_system

    Args
    ----
    t   := float
           value of the independent variable (time)
    m   := dimension of xyz
    xyz := np.array (shape: 3)
           values of the dependent variables at time t
    
    Returns
    -------
    dxdt := values of derivatives of the dependent 
            variables at time t
    """
    beta = 8.0 / 3.0
    rho = 28.0
    sigma = 10.0

    dxdt = np.zeros(3)

    dxdt[0] = sigma * (xyz[1] - xyz[0])
    dxdt[1] = xyz[0] * (rho - xyz[2]) - xyz[1]
    dxdt[2] = xyz[0] * xyz[1] - beta * xyz[2]

    return dxdt


def rk4vec(t0, m, u0, dt, f):
    """
    One Runge-Kutta step for a vector ODE.

    Args
    ----
    t0 := float
          current time
    m  := spatial dimension
    u0 := solution estimate at current timestep
    dt := the time step
    f  := u' = F(t,m,u)
          evaluates the derivative u' given time t 
          and the solution vector u
    
    Returns
    -------
    u  := the fourth order Runge-Kutta solution
          estimate at time t0+dt
    """
    # get 4 sample values of derivative

    f0 = f(t0, m, u0)

    t1 = t0 + dt / 2.0
    u1 = u0 + dt * f0 / 2.0
    f1 = f(t1, m, u1)

    t2 = t0 + dt / 2.0
    u2 = u0 + dt * f1 / 2.0
    f2 = f(t2, m, u2)

    t3 = t0 + dt
    u3 = u0 + dt * f2
    f3 = f(t1, m, u1)

    # runge-kutta linear approximation
    u = u0 + dt * (f0 + 2.0 * f1 + 2.0 * f2 + f3) / 6.0

    return u


def lorenz_ode_plot_3d(n, t, x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, linewidth=2, color='b')
    ax.grid(True)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('z(t)')
    plt.savefig('lorenz_ode_3d.png')
    plt.clf()


if __name__ == "__main__":
    import argparse
    parser.add_argument('--n', type=int, default=2000)
    args = parser.parse_args()
    
    t, x, y, z = lorenz_ode_compute(args.n, 40.0, 8.0, 1.0, 1.0)
    lorenz_ode_plot_3d(n, t, x, y, z)
