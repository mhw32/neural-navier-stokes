import numpy as np


def get_gauss_lobatto_points(N, k=1):
    i = np.arange(N)
    x_i = np.cos(k * np.pi * i / float(N - 1))
    return x_i


def get_bar_c_k(k, K):
    assert k >= 0
    return 2 if (k == 0 or k == K) else 1


def get_T_matrix(N, K):
    T = np.stack( [ get_gauss_lobatto_points(N, k=k)
                    for k in np.arange(K) ] ).T
    return T


def get_inv_T_matrix(N, K):
    inv_T = np.stack([np.cos(np.pi * np.arange(N) / float(N - 1))
                      for k in np.arange(K)]).T
    bar_c_i = np.stack([np.repeat(get_bar_c_k(i, K), N)
                        for i in np.arange(K)])
    bar_c_k = bar_c_i.T
    inv_T = 2 * inv_T / (bar_c_k * bar_c_i * N)
    return inv_T


data = np.load('../data/data_semi_implicit.npz')
U = data['u']
nt, nx, ny = u.shape[0], u.shape[1], u.shape[2]
n_coeff = 10

Tx = get_T_matrix(nx, n_coeff)
Ty = get_T_matrix(ny, n_coeff)
Tx_inv = get_inv_T_matrix(nx, n_coeff)
Ty_inv = get_inv_T_matrix(ny, n_coeff)

U_hat = (Tx_inv @ U) @ Ty_inv.T
U = (Tx @ U_hat) @ Ty.T
