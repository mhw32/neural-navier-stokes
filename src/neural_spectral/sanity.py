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
                      for k in np.arange(K)])
    bar_c_i = np.array([get_bar_c_k(i, N) for i in range(N)])[np.newaxis, :]
    bar_c_k = np.array([get_bar_c_k(k, K) for k in range(K)])[:, np.newaxis]

    inv_T = 2 * inv_T / (bar_c_k * bar_c_i * N)
    return inv_T


data = np.load('../data/data_semi_implicit.npz')
Us = data['u']
nt, nx, ny = Us.shape[0], Us.shape[1], Us.shape[2]
n_coeff = 51

Tx = get_T_matrix(nx, n_coeff)    # k x N
Ty = get_T_matrix(ny, n_coeff).T  # N x k
T = Tx @ Ty  # k x k
T_inv = np.linalg.inv(T)

for t in range(nt): 
    U = Us[t]
    U_hat = T @ U
    U_recon = T_inv @ U_hat
    print(np.linalg.norm(U - U_recon))
    break

